from jose import JWTError, jwt
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def create_access_token(config: dict, data: dict):
    """
    Création d'un token d'accès

    :param config: app config
    :param data: data à encoder
    :return: str, le jwt encodé
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=config['app']['security']['token_lifetime_in_min'])
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, 
                             config['app']['security']['token_secret_key'], 
                             algorithm=config['app']['security']['token_algorithm'])
    return encoded_jwt
