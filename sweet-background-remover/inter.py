import gettext

#symbols = "en", "pt-br"

def inter(symbol):
    # set current language
    lang_translations = gettext.translation('base', localedir='locales', languages=[symbol])
    lang_translations.install()
    # define _ shortcut for translations
    return lang_translations.gettext

