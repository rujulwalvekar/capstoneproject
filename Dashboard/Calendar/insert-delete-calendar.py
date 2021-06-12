from pprint import pprint
from Google import Create_Service

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'calendar'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/calendar']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

print(dir(service))

request_body = {
    'summary' : ' Yo my 1st calendar'
}

'''
To create a new calendar
'''
response = service.calendars().insert(body=request_body).execute()
pprint(response)

'''
To delete a new calendar
'''
# response = service.calendars().delete(calendarId='2v3e8tm0ndqohnn6n9cm974nvs@group.calendar.google.com').execute()
# pprint(response)
