from pprint import pprint
import Calendar.Google as Google
import os
from datetime import datetime


def get_events(calendarId):
    CLIENT_SECRET_FILE = "Calendar/client_secret.json"
    API_NAME = 'calendar'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    service = Google.Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    print("GService ", service)
    page_token = None
    while True:
        start_time = datetime.utcnow().replace(hour=0, minute=0).isoformat() + 'Z'
        end_time = datetime.utcnow().replace(hour=23, minute=59).isoformat() + 'Z'
        events = service.events().list(calendarId=calendarId, pageToken=page_token,
                                       timeMin=start_time,
                                       timeMax=end_time).execute()
        # pprint(events['items'][2]['attendees'][1]['responseStatus'])
        return events['items']  # Returns events of the day
