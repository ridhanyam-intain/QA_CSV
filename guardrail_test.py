import guardrails as gd
from guardrails.hub import RestrictToTopic
from guardrails.errors import ValidationError

valid_topics = ["bike"]
invalid_topics = ["phone", "tablet", "computer"]

text = """Introducing the Galaxy Tab S7, a sleek and sophisticated device that seamlessly combines \
cutting-edge technology with unparalleled design. With a stunning 5.1-inch Quad HD Super AMOLED display, \
every detail comes to life in vibrant clarity. The Samsung Galaxy S7 boasts a powerful processor, \
ensuring swift and responsive performance for all your tasks. \
Capture your most cherished moments with the advanced camera system, which delivers stunning photos in any lighting conditions."""


# Create the Guard with the OnTopic Validator
guard = gd.Guard.for_string(
    validators=[
        RestrictToTopic(
            valid_topics=valid_topics,
            invalid_topics=invalid_topics,
            device=-1,
            llm_callable="gpt-3.5-turbo",
            disable_classifier=False,
            disable_llm=False,
            on_fail="exception",
        )
    ],
)

# Test with a given text
try:
    guard.parse(
        llm_output=text,
    )
except ValidationError as e:
    print(e)


