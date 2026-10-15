"""tools/gpu/monitor_inspection.py 的取回函数 fetch() 的单测。

为什么值得单独测：fetch() 是产物回收链路上唯一真正搬字节的地方，而它的失败模式
只有在任务跑完那一刻才出现（远端 phase=='complete'）。这组用例把三类失败钉死在
本机：摘要算错、远端变短（size guard）、会话卡死（必须抛 OSError 而不是无限挂住）。
"""
import hashlib
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'gpu'))
import monitor_inspection as mi  # noqa: E402


class FakeChannel:
    def __init__(self):
        self.timeout = 'unset'

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value


class FakeSFTP:
    def __init__(self, payload, *, stall=False, remote_size=None):
        self.payload = payload
        self.stall = stall
        self.remote_size = remote_size
        self.channel = FakeChannel()
        self.getfo_calls = []

    def get_channel(self):
        return self.channel

    def getfo(self, remote, fl, callback=None, prefetch=None,
              max_concurrent_prefetch_requests=None):
        self.getfo_calls.append(
            {'remote': remote, 'prefetch': prefetch,
             'concurrency': max_concurrent_prefetch_requests})
        if self.stall:
            raise socket.timeout('simulated stall')
        fl.write(self.payload)

    def stat(self, remote):
        size = len(self.payload) if self.remote_size is None else self.remote_size
        return type('Stat', (), {'st_size': size})()


class FetchTests(unittest.TestCase):
    def test_digest_is_of_what_actually_lands(self):
        payload = b'x' * (3 * 1024 * 1024 + 17)
        sftp = FakeSFTP(payload)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'a.bin'
            digest = mi.fetch(sftp, '/remote/a.bin', dest)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse((Path(tmp) / 'a.bin.part').exists(),
                             '落地后不应残留 .part')

    def test_prefetch_requested_and_channel_timeout_restored(self):
        sftp = FakeSFTP(b'abc')
        with tempfile.TemporaryDirectory() as tmp:
            mi.fetch(sftp, '/remote/a.bin', Path(tmp) / 'a.bin')
        self.assertEqual(sftp.getfo_calls[0]['prefetch'], True,
                         '必须开 prefetch，否则跨境链路按往返时延传输')
        self.assertEqual(sftp.channel.timeout, 'unset',
                         '取回结束后必须恢复通道原有的超时设置')

    def test_remote_shorter_than_local_part_raises_and_keeps_part(self):
        payload = b'y' * 4096
        sftp = FakeSFTP(payload, remote_size=len(payload) - 10)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'a.bin'
            with self.assertRaises(RuntimeError) as ctx:
                mi.fetch(sftp, '/remote/a.bin', dest)
            self.assertIn('Recovery size mismatch', str(ctx.exception))
            self.assertFalse(dest.exists(), '校验没过就不许换名落地')
            self.assertTrue((Path(tmp) / 'a.bin.part').exists())

    def test_stall_raises_oserror_so_caller_can_retry(self):
        sftp = FakeSFTP(b'z' * 1024, stall=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'a.bin'
            with self.assertRaises(OSError):
                mi.fetch(sftp, '/remote/a.bin', dest)
            self.assertFalse(dest.exists())
        self.assertEqual(sftp.channel.timeout, 'unset',
                         '抛错路径也要恢复通道超时设置')


if __name__ == '__main__':
    unittest.main()
