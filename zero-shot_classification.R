
library(reticulate)
cat("torch module available? ", py_module_available("torch"), "\n")
cat("transformers module available? ", py_module_available("transformers"), "\n")


py_install("transformers", pip=TRUE)
py_install("torch", pip=TRUE)

py_require("transformers")
py_require("torch")

# Import the transformers Python library
transformers <- import("transformers")
torch <- import("torch")

print(torch$rand(tuple(3L, 3L)))
print(torch$`__version__`)

py_config()

# Create a zero-shot classification pipeline.
# This uses a pre-trained model that supports zero-shot tasks.
classifier <- transformers$pipeline(
  "zero-shot-classification",
  model = "facebook/bart-large-mnli",
  framework='pt'
)

# Define the text (sequence) to classify and candidate labels.
sequence <- "The economy is experiencing rapid growth and innovation."
candidate_labels <- c("economics", "politics", "technology", "sports")

# Perform zero-shot classification
result <- classifier(sequence, candidate_labels)

# Print the result
print(result)

