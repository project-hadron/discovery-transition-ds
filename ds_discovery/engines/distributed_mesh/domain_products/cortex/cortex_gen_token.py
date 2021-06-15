__author__ = 'Darryl Oatridge'

from cortex.utils import generate_token


PAT = {"jwk": {"crv": "Ed25519",
               "x": "OXdyU11SG10iRiaYststIz5sSt7Dk0qWd-AEdVW-CA0",
               "d": "HgCri9Xw33SC3qCQMG5Q1dcixv7OgU9lf91OCeiK7-g",
               "kty": "OKP",
               "kid": "tpmSp9SwgNUlZQC7xIe3wqwFPW6EcjTrn2hpfCnHjZ4"},
       "issuer": "cognitivescale.com",
       "audience": "cortex",
       "username": "f42d74b1-d67b-48dc-ab09-5d0771d6b7c0",
       "url": "https://api.dci-dev.dev-eks.insights.ai"}

print(generate_token(PAT))