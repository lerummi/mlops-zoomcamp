import datetime
from prefect import flow, task
from prefect_email import EmailServerCredentials, email_send_message


@task(log_prints=True)
def datetime_msg() -> str:
    now = datetime.datetime.now()

    msg = f"Current time is {now}!"

    print(msg)

    return msg


@flow
def email_send_message_flow():
    email_server_credentials = EmailServerCredentials.load("my-email-block")

    msg = datetime_msg()

    subject = email_send_message(
        email_server_credentials=email_server_credentials,
        subject="Prefect Flow Notification",
        msg=msg,
        email_to="martin.krause.85@gmx.de",
    )


if __name__ == "__main__":
    email_send_message_flow()
