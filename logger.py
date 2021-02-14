import json
import time
import sys
from sys import platform
import logging
import os
from logging.handlers import RotatingFileHandler


class No_Info_Log(logging.Filter):
    """Override the logging Filter to allow for a no info log"""
    def filter(self, record):
        return record.levelno != logging.INFO


class SocketHandler(logging.handlers.SocketHandler):
    """Override the SockerHandler becuase the base handle uses separatir and pickles."""

    def __init__(self, host, port, socket_callback):
        super().__init__(host, port)
        self.socket_callback = socket_callback

    def createSocket(self):
        """Here I have added the socket callback
            the callback notifies the Renderer and forces an the socket exit when finished.
        """
        now = time.time()
        # Either retryTime is None, in which case this
        # is the first time back after a disconnect, or
        # we've waited long enough.
        if self.retryTime is None:
            attempt = True
        else:
            attempt = (now >= self.retryTime)
        if attempt:
            try:
                self.sock = self.makeSocket()
                self.retryTime = None  # next time, no delay before trying
                self.socket_callback(self.sock)
            except OSError:
                # Creation failed, so set the retry time and return.
                if self.retryTime is None:
                    self.retryPeriod = self.retryStart
                else:
                    self.retryPeriod = self.retryPeriod * self.retryFactor
                    if self.retryPeriod > self.retryMax:
                        self.retryPeriod = self.retryMax
                self.retryTime = now + self.retryPeriod

    def makePickle(self, record):
        """PICKLE RICK!!! (is trash bc it was chunking things"""
        ei = record.exc_info
        if ei:
            self.format(record)
        d = dict(record.__dict__)
        d['msg'] = record.getMessage()
        d['args'] = None
        d['exc_info'] = None
        d.pop('message', None)
        d = json.dumps(d) + "\n"
        b = d.encode("UTF-8")
        return b


def create_logger(
    logger_name: str,
    tmp_folder: str,
    no_info: bool,
    silent: bool,
    socket_callback,
):
    """Get a single logger with following Handlers:
        StreamHandler (stdout) - this is to print to shell
        StreamHandler (stderr) - this is to print to error (useful for kill command)
        SocketHandler ('localhost', 42069) - this is for advanced logging to jumpcutter.
        RotatingFileHandler (render.log) - this keeps up to file log files
    """
    logging.basicConfig()
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    if not silent:
        stdout_logger = logging.StreamHandler(sys.stdout)
        stdout_logger.setLevel(logging.INFO)
        stdout_logger.setFormatter(logging.Formatter())
        stdout_logger.flush = sys.stdout.flush
        logger.addHandler(stdout_logger)

        stderr_logger = logging.StreamHandler(sys.stderr)
        stderr_logger.setLevel(logging.ERROR)
        stderr_logger.setFormatter(logging.Formatter())
        stderr_logger.flush = sys.stderr.flush
        logger.addHandler(stderr_logger)

    hostname = (os.path.join(tmp_folder, "render.sock"), None)
    if platform == "win32":
        hostname = ('localhost', 42069)
    socket_logger = SocketHandler(*hostname, socket_callback)
    # socket_logger = logging.handlers.SocketHandler(*hostname)
    # socket_logger = logging.handlers.DatagramHandler(*hostname)
    socket_logger.setLevel(logging.INFO)
    socket_logger.setFormatter(logging.Formatter())
    logger.addHandler(socket_logger)
    log_folder = os.path.join(tmp_folder, "../logs")
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    file_logger = RotatingFileHandler(
        os.path.join(log_folder, "./render.log"),
        maxBytes=5*1000*1024, backupCount=5, encoding=None, delay=False
        # the discord max is 8 MB
    )
    file_logger.setLevel(logging.DEBUG)
    file_logger.setFormatter(logging.Formatter(
        '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    ))
    if no_info:
        file_logger.addFilter(No_Info_Log())
    logger.addHandler(file_logger)
    # assert(False), sock
    return logger
    # return logger, sock
