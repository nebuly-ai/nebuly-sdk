from __future__ import annotations

import atexit
from queue import Empty, Queue
from threading import Thread

from nebuly.entities import Message, Publisher_T


class ConsumerWorker:
    def __init__(self, queue: Queue[Message], publish: Publisher_T) -> None:
        self.publish = publish
        self.queue: Queue[Message] = queue
        self.running = True
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def run(self) -> None:
        while self.running:
            try:
                message = self.queue.get(timeout=0.1)
            except Empty:
                continue
            self.publish(message)
            self.queue.task_done()

    def stop(self) -> None:
        self.queue.join()
        self.running = False
        self.thread.join()
