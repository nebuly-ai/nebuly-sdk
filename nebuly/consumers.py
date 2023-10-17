from __future__ import annotations

import atexit
from queue import Empty, Queue
from threading import Thread

from nebuly.entities import InteractionWatch, Publisher


class ConsumerWorker:
    def __init__(self, queue: Queue[InteractionWatch], publish: Publisher) -> None:
        self.publish = publish
        self.queue: Queue[InteractionWatch] = queue
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
