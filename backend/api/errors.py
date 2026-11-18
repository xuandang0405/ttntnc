class AppError(Exception):
    def __init__(self, message: str, code: str = 'INVALID_ACTION', status: int = 400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status
