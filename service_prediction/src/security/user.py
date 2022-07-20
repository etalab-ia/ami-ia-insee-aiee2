from passlib.apps import custom_app_context as pwd_context
import json


class User:

    def __init__(self, username, hashed_pwd=None):
        """
        Classe de gestion des utilisateurs

        :param username: nom d'utilisateur
        :param hashed_pwd: str, password hashé
        """
        self.username = username
        self.password_hash = hashed_pwd

    def hash_password(self, password):
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)


def load_users(config):
    """
    Chargement des utilisateurs depuis le fichier password et 
    la config / les variables d'env surchargeant la config

    :param config: app config
    :return: dict {username: hashed_pwd}
    """
    password_file = config['app']['security']['password_file']
    try:
        with open(password_file) as f:
            passwords = json.load(f)
    except FileNotFoundError:
        passwords = {}
    if config['app']['security']['username'] and config['app']['security']['password']:
        user = User(config['app']['security']['username'])
        user.hash_password(config['app']['security']['password'])
        passwords[user.username] = user.password_hash
        with open(password_file, 'w') as f:
            json.dump(passwords, f)

    return passwords


def verify_user(users, username, password):
    """
    Vérification d'un utilisateur

    :param users: dict {username: hashed_pwd}
    :param username: username à vérifier
    :param password: password non hashé à vérifier
    :return: False ou username
    """
    if username not in users:
        return False
    user = User(username, users[username])
    if not user.verify_password(password):
        return False
    return username