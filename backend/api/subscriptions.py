import stripe
import os
import logging
from dotenv import load_dotenv

load_dotenv() 

logging.getLogger("stripe").setLevel(logging.WARNING)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def create_subscription(email, price_id):
    customer = stripe.Customer.create(email=email)

    subscription = stripe.Subscription.create(
        customer=customer.id,
        items=[{"price": price_id}],
        payment_behavior="default_incomplete",
        expand=["latest_invoice.payment_intent"],
    )

    payment_intent = subscription.latest_invoice.payment_intent


    logger.info("Subscription created", extra={
        "subscription_id": subscription.id,
        "customer_id": subscription.customer,
        "payment_intent_id": payment_intent.id,
        "status": payment_intent.status
    })


    return {
        "subscriptionId": subscription.id,
        "clientSecret": subscription.latest_invoice.payment_intent.client_secret,
        "publishableKey": os.getenv("STRIPE_PUBLISHABLE_KEY")
    }
