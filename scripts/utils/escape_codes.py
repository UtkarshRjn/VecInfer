BLUE = '\033[94m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'

END = '\033[0m'


def escape_code_factory(CODE):
    def escape_code(*args, **kwargs):
        print(CODE, end='')
    return escape_code