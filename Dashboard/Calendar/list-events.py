from pprint import pprint
from Google import Create_Service

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'calendar'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/calendar']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
page_token = None
while True:
  events = service.events().list(calendarId='primary', pageToken=page_token).execute()
  pprint(events['items'][0])
  for event in events['items']:
    pprint(event.get('summary'))
  page_token = events.get('nextPageToken')
  if not page_token:
    break