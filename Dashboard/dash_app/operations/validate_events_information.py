from dash_app.datetime_utilities import datetime_utilities


class ValidateEventsInformation(object):
    def __init__(self, events):
        self.events = events

    def convert_event_datetime_to_military_time(self):
        for event in self.events:
            event_start = event['start']['dateTime']
            event_end = event['end']['dateTime']
            event_start_datetime = datetime_utilities.convert_str_datetime_to_datetime_obj(event_start.split('+')[0])
            event_end_datetime = datetime_utilities.convert_str_datetime_to_datetime_obj(event_end.split('+')[0])

            event['start'] = datetime_utilities.get_military_time_from_datetime(event_start_datetime)
            event['end'] = datetime_utilities.get_military_time_from_datetime(event_end_datetime)
            print('local_start', event)

        return self.events
