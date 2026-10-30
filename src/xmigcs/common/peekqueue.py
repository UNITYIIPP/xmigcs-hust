import queue


class PeekableQueue(queue.Queue):
    def peek(self):
        """Compatibility peek; prefer :meth:`get_latest` in real-time paths."""
        with self.mutex:
            if not self._qsize():
                raise queue.Empty
            return self.queue[0]

    def put_latest(self, item):
        """Atomically replace the buffered value with the newest item."""
        with self.not_full:
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                self._get()
                if self.unfinished_tasks:
                    self.unfinished_tasks -= 1
                self.not_empty.notify()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get_latest(self):
        """Return and remove the newest buffered value without blocking.

        Older samples are deliberately discarded: control uses the newest
        available state rather than replaying stale samples.
        """
        with self.not_empty:
            if not self._qsize():
                raise queue.Empty
            item = self._get()
            if self.unfinished_tasks:
                self.unfinished_tasks -= 1
            while self._qsize():
                item = self._get()
                if self.unfinished_tasks:
                    self.unfinished_tasks -= 1
            self.not_full.notify_all()
            return item
