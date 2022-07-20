import os
import logging
import logging.config
import shutil

from backend_utils.config_parser import get_local_file, parse_full_config

# Chargement de la config
config = parse_full_config(get_local_file('config.yaml'), get_local_file('config_env.yaml'))
logging.config.fileConfig(get_local_file('logging.ini'))
logging.getLogger().setLevel(config['app']['log_level'])

from fastapi import Depends, FastAPI, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from security.user import load_users, verify_user
from security.token import Token, TokenData, create_access_token

from app_prediction import Models, BIModelRequest, BISaveRequest

logger = logging.getLogger('PredictionService')
logger.debug('Creating app')

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()

###################
# User management
###################

users = load_users(config)


def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validation du token et extraction de l'utilisateur depuis le token

    :param token: le token d'identification
    :return: le username
    :raise: credentials_exception
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, 
                             config['app']['security']['token_secret_key'], 
                             algorithms=[config['app']['security']['token_algorithm']])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data.username


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint de récupération d'un token

    :param form_data: data json contenant 'username' et 'password'
    :return: {"access_token": access_token, "token_type": "bearer"}
    """
    username = verify_user(users, form_data.username, form_data.password)
    if not username:
        logging.info(f'/token - credential errors for user {form_data.username}')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(config, {"sub": username})
    logging.info(f'/token - token emitted for user {form_data.username}')
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
def read_users_me(current_user: str = Depends(get_current_user)):
    """
    Test endpoint de contrôle de l'utilisateur par token

    :return: {'login': current_user}
    """
    logging.info(f'/users/me - {current_user}')
    return {'login': current_user}

@app.get("/")
def root():
    return {"message": "Hello World"}


####################
# Predictions
####################

# Chargement des modèles
try:
    models = Models(config)
    models.load_all_models()
except RuntimeError:
    exit(-1)


@app.get("/noncodable")
def handle_noncodable(request_data: BIModelRequest, 
                            current_user: str = Depends(get_current_user)):
    """
    Endpoint de prédiction non-codable
    Protégé par token

    :param request_data: BIModelRequest
    :param current_user: porté par le token lors de l'appel
    :return: {'noncodable': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'non_codable': bool, 'predictions': [score_0, score_1]}
    """
    logger.info(f'Detect NonCodable on cabbis {[d.cabbi for d in request_data.documents]}')
    try:
        return {'noncodable': models.predict_noncodable(request_data)}
    except Exception as e:
        logger.error(f'Error calculating noncodables for cabbis {[d.cabbi for d in request_data.documents]}')
        logger.error(f'Error : {e.__class__.__name__} - {e}')
        raise e


@app.get("/naf")
def handle_naf(request_data: BIModelRequest, 
                      current_user: str = Depends(get_current_user)):
    """
    Endpoint de prédiction naf
    Protégé par token

    :param request_data: BIModelRequest
    :param current_user: porté par le token lors de l'appel
    :return: {'naf': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
    """
    logger.info(f'predict NAF on cabbis {[d.cabbi for d in request_data.documents]}')
    try:
        return {'naf': models.predict_naf(request_data)}
    except Exception as e:
        logger.error(f'Error calculating naf for cabbis {[d.cabbi for d in request_data.documents]}')
        logger.error(f'Error : {e.__class__.__name__} - {e}')
        raise e


@app.get("/pcs")
def handle_pcs(request_data: BIModelRequest, 
                      current_user: str = Depends(get_current_user)):
    """
    Endpoint de prédiction pcs
    Protégé par token

    :param request_data: BIModelRequest
    :param current_user: porté par le token lors de l'appel
    :return: {'pcs': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
    """
    logger.info(f'predict PCS on cabbis {[d.cabbi for d in request_data.documents]}')
    try:
        return {'pcs': models.predict_pcs(request_data)}
    except Exception as e:
        logger.error(f'Error calculating pcs for cabbis {[d.cabbi for d in request_data.documents]}')
        logger.error(f'Error : {e.__class__.__name__} - {e}')
        raise e

@app.get("/siret")
def handle_siret(request_data: BIModelRequest, 
                        current_user: str = Depends(get_current_user)):
    """
    Endpoint de prédiction siret
    Protégé par token

    :param request_data: BIModelRequest
    :param current_user: porté par le token lors de l'appel
    :return: {'siret': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
    """
    logger.info(f'predict SIRET on cabbis {[d.cabbi for d in request_data.documents]}')
    
    try:
        return {'siret': models.predict_siret(request_data)}
    except Exception as e:
        logger.error(f'Error calculating Siret for cabbis {[d.cabbi for d in request_data.documents]}')
        logger.error(f'Error : {e.__class__.__name__} - {e}')
        raise e

@app.get("/save_bi")
def handle_save(request_data: BISaveRequest, 
                      current_user: str = Depends(get_current_user)):
    """
    Endpoint de sauvegarde de BIs
    Cette fonction est appelée lorsque le gestionnaire a validé un BI
    Elle récupère les données validées dans la base et les push dans les sources 
    de données pour rendre le BI disponible aux futures requètes

    Protégé par token

    :param request_data: BIModelRequest
    :param current_user: porté par le token lors de l'appel
    :return: {'save_bi': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
    """
    logger.info(f'saving BIs with cabbis {[d.cabbi for d in request_data.documents]}')
    try:
        return {'save_bi': models.push_bi_to_datasources(request_data)}
    except Exception as e:
        logger.error(f'Error saving BIs with cabbis {[d.cabbi for d in request_data.documents]}')
        logger.error(f'Error : {e.__class__.__name__} - {e}')
        raise e

logger.info('App started')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)