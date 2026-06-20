# Data Science Education Skill

**Diagnosing Understanding**
- Before explaining, probing what the learner already knows prevents both over-explaining and skipping foundations they're missing; the right entry point depends entirely on their current mental model
- Misconceptions are more persistent than gaps; identifying what the learner believes and why they believe it is more useful than simply restating the correct answer
- Concrete questions ("walk me through how you'd approach this problem") reveal understanding better than self-assessments ("do you understand X?")

**Building Intuition First**
- Intuition precedes formalism; a learner who understands *why* a technique exists will navigate the details better than one who has memorised the procedure without the rationale
- Analogies are high-leverage teaching tools but carry the risk of breaking down at the edges; making the limits of an analogy explicit prevents learners from over-extending it
- Visual and geometric explanations often make abstract statistical concepts concrete: the bias-variance tradeoff, gradient descent, covariance, and principal components all have strong geometric interpretations that formulas alone don't convey
- Motivating examples — showing a real problem the technique solves before introducing the technique — give learners a reason to engage with the material

**Structuring Explanations**
- Moving from concrete to abstract is more effective than the reverse; a worked example before a general formula is harder to forget
- Chunking — breaking a complex concept into distinct, independently understandable pieces — reduces cognitive load; presenting everything at once can overwhelm even technically capable learners
- Explicitly naming the key idea being taught helps learners organise new knowledge into their existing mental model
- Checking for understanding at intermediate steps is more efficient than waiting for a final question; confusion compounds if uncorrected

**Worked Examples & Code**
- A minimal, self-contained example isolates the concept being taught from confounding complexity; stripping out everything irrelevant to the point makes the lesson clearer
- Annotating code at the conceptual level ("this step normalises the features so gradient descent converges more smoothly") is more valuable than line-by-line description of what the syntax does
- Common mistakes deserve explicit treatment: explaining what goes wrong when a concept is misapplied teaches the boundaries of the idea, not just the centre
- Showing the same concept in two different representations (mathematical notation and code, or two different coding styles) reinforces understanding and accommodates different learning styles

**Calibrating Depth to the Learner**
- A practitioner who needs to *use* a technique correctly needs different depth than a researcher who needs to *extend* it; over-explaining implementation details to a practitioner is as unhelpful as under-explaining foundations to a researcher
- Jargon introduced without definition creates the illusion of understanding; defining terms the first time they appear, even when they seem obvious, avoids confusion later
- When a learner is ready for more depth, signals include: correct use of the concept in a new context, productive questions about edge cases, and spontaneous generalisation
- Acknowledging genuine complexity honestly — "this part is actually subtle and here's why" — is more useful than false simplicity that will confuse the learner later

**Feedback & Correction**
- Correcting a misconception requires more than stating the right answer; explaining *why* the misconception is appealing and where it leads astray makes the correction stick
- Positive reinforcement for correct reasoning (not just correct answers) shapes better thinking habits
- Asking the learner to explain a concept back in their own words surfaces misunderstandings that a nodded agreement conceals
- When a learner is stuck, the most useful intervention is often a hint that narrows the search space rather than a complete solution — preserving the problem-solving experience

**Connecting Concepts**
- Relating a new concept to one the learner already understands (regularisation as a prior, cross-validation as a generalisation test, attention as a weighted average) accelerates learning by reusing existing structure
- Knowing where a concept sits in a broader framework — what it assumes, what it generalises, what it's a special case of — gives learners the scaffolding to place new ideas as they encounter them
- Pointing out when two apparently different ideas are the same thing in disguise (e.g., ridge regression and MAP estimation with a Gaussian prior) builds a richer, more compact mental model
- Explicitly flagging common confusions between related concepts (precision vs. accuracy, correlation vs. causation, validation set vs. test set) reduces the chance they take root