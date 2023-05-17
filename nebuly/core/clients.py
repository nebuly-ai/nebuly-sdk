from queue import Empty
from threading import Thread

from nebuly.core.queues import NebulyQueue
from nebuly.core.schemas import NebulyDataPackage
from nebuly.utils.logger import nebuly_logger


class NebulyClient:
    def send_request_to_nebuly_server(self, request_data: NebulyDataPackage):
        nebuly_logger.info(f"\n\nDetected Data:\n {request_data.json()}")


class NebulyTrackingDataThread(Thread):
    # TODO: add a way to stop the thread.
    # If i define the thread as deamon it will stop when the main thread stops
    # and this can create unexpected behaviour.
    # i.e. if i am processing an audio track what happens is:
    # the main finish the execution before the audio track is processed
    # and the thread gets killed before issuing the request to the server

    def __init__(
        self,
        queue: NebulyQueue,
        nebuly_client: NebulyClient = NebulyClient(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._queue = queue
        self._nebuly_client = nebuly_client
        self.thread_running = True

    def run(self):
        while self.thread_running:
            try:
                queue_object = self._queue.get()
            except Empty:
                continue
            except KeyboardInterrupt:
                # what happens if there is a keyboard interrupt?
                # I need to save the status of the queue and load it
                # back when it is started again
                return

            request_data = queue_object.as_data_package()
            self._nebuly_client.send_request_to_nebuly_server(request_data)
            self._queue.task_done()
