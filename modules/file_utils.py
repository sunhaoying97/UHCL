"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import logging
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps

from tqdm import tqdm

import boto3
from botocore.exceptions import ClientError
import requests

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))


#ETag（实体标签）是 HTTP 响应头中的一个标识符，用于表示特定版本的资源。它可以让缓存更高效，节省带宽
# 因为如果内容没有改变，Web 服务器不需要重新发送完整的响应。此外，ETags 还可以帮助防止资源的同时更新互相覆盖。
# 如果给定 URL 的资源发生变化，必须生成新的 ETag 值。可以通过比较它们来确定两个资源的表示是否相同。


def url_to_filename(url: str, etag: str = None) -> str:
    #将 URL 转换为一个可重复的哈希文件名。如果指定了 etag，则将其哈希值附加到 URL 的哈希值后面，两者之间用一个点分隔
    
    #函数的主要作用是将给定的 URL 转换为一个哈希文件名，这个过程是可重复的，也就是说，对于同一个 URL，无论何时调用这个函数，都会得到相同的文件名。
    #这样做的原因主要有两个：
    #唯一性：通过将 URL 转换为其哈希值，可以确保每个 URL 都有一个唯一对应的文件名。这对于避免文件名冲突非常有用，特别是当我们需要从许多不同的 URL 下载文件并将它们存储在同一个目录中时。
    #一致性：哈希函数是确定性的，这意味着对于同一个输入，无论何时何地运行哈希函数，都会得到相同的输出。这种一致性使得我们可以通过 URL 来查找或引用已下载的文件，而无需每次都从网络上获取文件。


    #如果提供了 etag（一个 HTTP 响应头，通常用于缓存验证），则将其哈希值附加到 URL 的哈希值后面。这样做可以进一步确保文件名的唯一性，
    #因为即使两个 URL 相同，但如果它们的 etag 不同（可能因为文件已经更新），那么它们也会被赋予不同的文件名。这对于正确地处理文件更新非常有用
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8') #将 URL 编码为 UTF-8 字节串。
    url_hash = sha256(url_bytes) #使用 SHA-256 算法对 URL 字节串进行哈希运算。
    filename = url_hash.hexdigest() #将哈希值转换为十六进制字符串，作为文件名。

    #如果提供了 etag，则将其编码为 UTF-8 字节串，并使用 SHA-256 算法进行哈希运算。
    # 然后，将 etag 的哈希值（转换为十六进制字符串后）附加到文件名后面，两者之间用一个点分隔。
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename #返回生成的文件名。


def filename_to_url(filename: str, cache_dir: Union[str, Path] = None) -> Tuple[str, str]:
    #该函数返回为给定文件名存储的 URL 和 etag（可能为 None）。如果文件名或其存储的元数据不存在，则抛出 FileNotFoundError 异常
    #从给定的文件名中恢复原始的 URL 和 etag。
    # 当我们从网络下载文件并将其存储在本地时，通常会将文件的源 URL 和 etag（如果有）一起存储下来，以便于以后的引用或验证。 #这对于管理和维护本地缓存的文件非常有用。
    
    #etag 是 HTTP 响应头中的一个字段，通常用于缓存验证。通过比较本地存储的 etag 和服务器上文件的当前 etag，
    # 我们可以知道文件是否已经被修改或更新。如果 etag 不同，那么文件可能已经被更新，我们可能需要重新下载文件
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.
    """
    #如果没有提供 cache_dir，那么它会使用默认的缓存目录 PYTORCH_PRETRAINED_BERT_CACHE。
    # 如果 cache_dir 是 Path 对象，那么它会将其转换为字符串。
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    #构造缓存路径，该路径是将文件名添加到缓存目录后的路径。
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path): #如果缓存路径不存在，那么它会抛出一个 FileNotFoundError 异常。
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json' #构造元数据文件的路径，该路径是在缓存路径后面添加 ‘.json’。
    if not os.path.exists(meta_path): #如果元数据文件的路径不存在，那么它也会抛出一个 FileNotFoundError 异常。
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file: #打开并读取元数据文件，然后加载 JSON 数据。
        metadata = json.load(meta_file)
    url = metadata['url'] 
    etag = metadata['etag']
    #从元数据中获取 URL 和 etag。 
    # #通过读取文件的元数据（通常存储在一个与文件同名但扩展名为 ‘.json’ 的文件中），并返回存储的 URL 和 etag 
    return url, etag


def cached_path(url_or_filename: Union[str, Path], cache_dir: Union[str, Path] = None) -> str:
    #接受一个可能是 URL 或本地路径的输入，并确定其类型。如果输入是 URL，那么它会下载并缓存该文件，并返回缓存文件的路径。
    # 如果输入已经是一个本地路径，那么它会确保文件存在，然后返回该路径
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    #如果没有提供 cache_dir，那么它会使用默认的缓存目录 PYTORCH_PRETRAINED_BERT_CACHE
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    #如果 url_or_filename 或 cache_dir 是 Path 对象，那么它会将它们转换为字符串。
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    #使用 urlparse 函数解析 url_or_filename。
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
        #如果解析出的方案（scheme）是 ‘http’、‘https’ 或 ‘s3’，那么它会认为 url_or_filename 是一个 URL。
        # 在这种情况下，它会调用 get_from_cache 函数从缓存中获取文件（如果需要，会下载文件），并返回缓存文件的路径。
    elif os.path.exists(url_or_filename):#如果 url_or_filename 是一个存在的文件，那么它会直接返回该路径。
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        #如果解析出的方案是空字符串，那么它会认为 url_or_filename 是一个文件路径，但文件不存在。
        #在这种情况下，它会抛出一个 FileNotFoundError 异常
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:#如果都不是以上情况，那么它会抛出一个 ValueError 异常，因为它无法将 url_or_filename 解析为 URL 或本地路径。
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))
    
################################
#URL 以 “s3://” 开头表示这是一个 Amazon S3（Simple Storage Service）的资源定位符。
# Amazon S3 是 Amazon Web Services（AWS）提供的一种对象存储服务，用于存储和检索任意数量的数据。
#在 “s3://” URL 中，“s3” 是协议部分，表示这是一个 S3 资源，后面跟着的是存储桶名称和对象键。
# 例如，“s3://my-bucket/my-object” 表示在 “my-bucket” 存储桶中名为 “my-object” 的对象。
#此外，S3 还支持通过 HTTP 或 HTTPS 协议访问，这时 URL 的格式会稍有不同。
# 例如，“https://my-bucket.s3.region.amazonaws.com/my-object” 和 “https://s3.region.amazonaws.com/my-bucket/my-object” 都可以用于访问同一个对象1。
##################################

def split_s3_path(url: str) -> Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""#该函数将一个完整的 S3 路径分割为存储桶名称和路径。
    parsed = urlparse(url) #使用 urlparse 函数解析输入的 URL。
    if not parsed.netloc or not parsed.path: #如果解析出的网络位置（netloc）或路径（path）为空，那么它会抛出一个 ValueError 异常
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc #将解析出的网络位置赋值给 bucket_name，这是 S3 存储桶的名称。
    s3_path = parsed.path #将解析出的路径赋值给 s3_path，这是在 S3 存储桶中的路径。
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"): #如果 s3_path 以 “/” 开头，那么它会移除这个开头的 “/”。
        s3_path = s3_path[1:]
    return bucket_name, s3_path #返回存储桶名称和路径。


def s3_request(func: Callable):
    #是一个装饰器函数，它接受一个函数 func 作为参数。这个 func 应该是一个执行 S3 请求的函数。
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """
    #wrapper 函数是 s3_request 的内部函数，它接受一个 URL 和任意数量的其他参数。这个函数会尝试调用 func，并将所有参数传递给它。
    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try: #如果在调用 func 时抛出了 ClientError 异常，那么它会检查异常的错误代码。
             #如果错误代码是 404，那么它会抛出一个 FileNotFoundError 异常，并指出文件未找到。否则，它会直接抛出原始的 ClientError 异常。
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise
    #函数最后返回 wrapper 函数，这样就可以用它来装饰其他执行 S3 请求的函数，以提供更有帮助的错误消息
    return wrapper


@s3_request
def s3_etag(url: str) -> Optional[str]:
    #用于检查 S3 对象的 ETag。这个函数被 s3_request 装饰器装饰，以便在发生错误时提供更有帮助的错误消息。
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3") #创建一个 S3 资源对象，用于与 Amazon S3 服务进行交互。
    bucket_name, s3_path = split_s3_path(url) #将输入的 URL 分割为存储桶名称和 S3 路径。
    s3_object = s3_resource.Object(bucket_name, s3_path) #使用存储桶名称和 S3 路径创建一个 S3 对象。
    return s3_object.e_tag #返回 S3 对象的 ETag。


@s3_request
def s3_get(url: str, temp_file: IO) -> None: 
    #用于直接从 S3 下载文件
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3") #创建一个 S3 资源对象，用于与 Amazon S3 服务进行交互。
    bucket_name, s3_path = split_s3_path(url) #将输入的 URL 分割为存储桶名称和 S3 路径。
    #从 S3 存储桶中下载文件。这个方法接受两个参数：S3 路径和一个临时文件对象。文件将被下载到这个临时文件中
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url: str, temp_file: IO) -> None:
    #用于从给定的 URL 下载文件
    """Get method."""
    req = requests.get(url, stream=True) #发送一个 GET 请求到 URL，并设置 stream=True 以允许流式下载
    content_length = req.headers.get('Content-Length') #从响应头中获取 ‘Content-Length’ 字段，这个字段表示了文件的总字节数。
    #如果 ‘Content-Length’ 字段存在，那么将其转换为整数并赋值给 total。否则，total 为 None。
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total) #创建一个 tqdm 进度条，单位为 “B”（字节），总量为 total

    #使用 req.iter_content 方法迭代响应内容。这个方法会返回一个迭代器，每次迭代都会返回一块内容（默认大小为 1KB）。
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk)) #对于每一块内容，如果它不为空（过滤掉保持连接的新块），那么就更新进度条，并将这块内容写入临时文件
            temp_file.write(chunk)
    progress.close() #下载完成后，关闭进度条


def get_from_cache(url: str, cache_dir: Union[str, Path] = None) -> str:
    #函数接受一个 URL，并在本地缓存中查找对应的数据集。如果数据集不在缓存中，那么它会下载该数据集。然后返回缓存文件的路径。
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    #如果没有提供 cache_dir，那么它会使用默认的缓存目录 PYTORCH_PRETRAINED_BERT_CACHE。
    # 如果 cache_dir 是 Path 对象，那么它会将其转换为字符串。
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    # Get eTag to add to filename, if it exists.
    # #如果 URL 是以 “s3://” 开头的，那么它会调用 s3_etag 函数来获取 eTag。
    # 否则，它会发送一个 HEAD 请求到 URL，并从响应头中获取 eTag。
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag) #调用 url_to_filename 函数，将 URL 和 eTag 转换为文件名

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename) #构造缓存路径，该路径是将文件名添加到缓存目录后的路径。

    if not os.path.exists(cache_path):
        #如果缓存路径不存在，那么它会下载文件到一个临时文件，然后在下载完成后将其复制到缓存目录。
        #这样做是为了防止在下载被中断时产生损坏的缓存条目。
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)
    #会返回缓存文件的路径。
    return cache_path


def read_set_from_file(filename: str) -> Set[str]:
    #从文件中提取一个去重的集合（set
    #预期的文件格式是每行一个项目。
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set() #创建一个空集合
    with open(filename, 'r', encoding='utf-8') as file_: #使用 ‘r’ 模式和 ‘utf-8’ 编码打开文件
        for line in file_:
            collection.add(line.rstrip()) #于文件中的每一行，去除行尾的空白字符，并将其添加到集合中。
    return collection


def get_file_extension(path: str, dot=True, lower: bool = True):#返回文件的扩展名
    """Return fie extension."""
    ext = os.path.splitext(path)[1] # 函数分割路径，获取扩展名
    ext = ext if dot else ext[1:] #则去掉扩展名前面的点。
    return ext.lower() if lower else ext #扩展名转换为小写并返回
