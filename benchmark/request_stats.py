from typing import Optional


class RequestStats:
    """
    Statistics collected for a particular AOAI request.
    """
    def __init__(self):
        self.request_start_time: Optional[float] = None
        self.response_status_code: int = 0
        self.response_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.response_end_time: Optional[float] = None
        self.context_tokens: int = 0
        self.generated_tokens: Optional[int] = None
        self.deployment_utilization: Optional[float] = None
        self.calls: int = 0
        self.last_exception: Optional[Exception] = None
        self.generated_text: Optional[str] = None
    
    def __str__(self) -> str:
        return f"RequestStats(request_start_time={self.request_start_time}, response_status_code={self.response_status_code}, response_time={self.response_time}, first_token_time={self.first_token_time}, response_end_time={self.response_end_time}, context_tokens={self.context_tokens}, generated_tokens={self.generated_tokens}, deployment_utilization={self.deployment_utilization}, calls={self.calls}, last_exception={self.last_exception}, generated_text={self.generated_text})"

    def __repr__(self) -> str:
        return f"RequestStats(\n" \
               f"    request_start_time: {self.request_start_time}\n" \
               f"    response_status_code: {self.response_status_code}\n" \
               f"    response_time: {self.response_time}\n" \
               f"    first_token_time: {self.first_token_time}\n" \
               f"    response_end_time: {self.response_end_time}\n" \
               f"    context_tokens: {self.context_tokens}\n" \
               f"    generated_tokens: {self.generated_tokens}\n" \
               f"    deployment_utilization: {self.deployment_utilization}\n" \
               f"    calls: {self.calls}\n" \
               f"    last_exception: {self.last_exception}\n" \
               f"    generated_text: {self.generated_text}\n" \
               f")"
