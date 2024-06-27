import os
from azure.identity import ManagedIdentityCredential, AzureCliCredential

TOKEN_SCOPE = "https://ml.azure.com"
def _get_credential():
    identity_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
    if identity_client_id is not None:
        credential = ManagedIdentityCredential(client_id=identity_client_id)
    else:
        credential = AzureCliCredential()
    return credential

def _token():
    credential = _get_credential()
    __token = credential.get_token(TOKEN_SCOPE)
    return __token.token

if __name__ == "__main__":
    print(_token())