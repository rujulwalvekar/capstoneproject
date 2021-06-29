from datetime import datetime

import pytz

TIMEZONE = pytz.timezone('Asia/Kolkata')


def get_current_datetime():
    return datetime.utcnow()


def convert_str_datetime_to_datetime_obj(datetime_str, parser='%Y-%m-%dT%H:%M:%S'):
    print(datetime_str)
    return datetime.strptime(datetime_str, parser)


def get_military_time_from_datetime(datetime, parser='%H:%M'):
    return datetime.strftime(parser)


def convert_utc_to_local(utc_datetime):
    return datetime.fromtimestamp(utc_datetime)


def convert_datetime_to_epoch(datetime_str, parser='%Y-%m-%dT%H:%M:%S.%f'):
    product_available_from_datetime = datetime.strptime(datetime_str.split('+')[0],
                                                        parser)
    epoch_start_datetime = datetime(1970, 1, 1)
    return (product_available_from_datetime - epoch_start_datetime).total_seconds()
