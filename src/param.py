#!/usr/bin/env python

""" Param """


class Param:
    """
    Provide a parameter that can be modified by key strokes
      `value` is the initial value of the parameter
      `dn_keys` are the registered keys that will decrease the `value` by `step`
      `up_keys` are the registered keys that will increase the `value` by `step`
      `value` cannot be adjusted beyond `minimum` and `maximum`
      if `wrap` is set,
        adjusted `value` > `maximum` will be set to `minimum` and
        adjusted `value` < `minimum` will be set to `maximum`
    """

    registered = []

    def __init__(
        self, value, dn_keys, up_keys, minimum=0, maximum=1, step=1, wrap=False
    ):
        self.value = value
        self.dn_keys = [ord(key) if isinstance(key, str) else key for key in dn_keys]
        self.up_keys = [ord(key) if isinstance(key, str) else key for key in up_keys]
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.wrap = wrap
        if wrap:
            assert minimum is not None and maximum is not None
        Param.registered.append(self)

    def adjust(self, key: int) -> bool:
        """
        Adjust value according to the key.
        Return whether the value was adjusted
        """
        if key in self.dn_keys:
            self.value -= self.step
            if self.minimum is not None and self.value < self.minimum:
                self.value = self.maximum if self.wrap else self.minimum
            return True

        if key in self.up_keys:
            self.value += self.step
            if self.maximum is not None and self.value > self.maximum:
                self.value = self.minimum if self.wrap else self.maximum
            return True
        return False

    @classmethod
    def handle(cls, key: int) -> bool:
        """
        Adjust parameters given a key event.
        Return whether the value was adjusted
        """
        return any(param.adjust(key) for param in cls.registered)
