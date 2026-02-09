# Progress Report: PopSign V3 - Debugging Results & New Direction

Following my last update about the configuration mismatch in my testing script, I have completed debugging and conducted systematic testing. Here are my findings:

## Debugging Results

**Issue Resolved:** The configuration mismatch has been fixed. The model weights, architecture, and label maps are now correctly aligned. Testing confirmed:
- Weights: 18 arrays matching the 13-class LSTM architecture
- Label map: 13 signs correctly mapped
- Model predictions: Sum to 1.0, functioning correctly

**Conclusion:** The technical setup is working correctly. The performance gap is not a bug it reflects genuine model limitations.

## Systematic Real-World Testing (13 Signs)

I conducted controlled testing on all 13 signs, performing each sign multiple times:

| Status | Signs | Real-World Performance |
|--------|-------|------------------------|
|  Works well | arm (99.2%), hello (97.1%), apple (89%) | 3 signs |
|  Works with low confidence | TV (27.1%), animal (57.6%), aunt (58%), awake (41%) | 4 signs |
|  Does not work | after, airplane, all, alligator, another, any | 6 signs |

**Real-world accuracy: 7/13 signs = 54%**

### Confusion Patterns Discovered

Signs that fail get consistently confused with similar signs:

| Actual Sign | Predicted As | Reason |
|-------------|--------------|--------|
| after | arm (97.8%) | Similar arm movement |
| airplane | aunt/awake | Hand near face |
| all | any (59.6%) | Very similar hand shapes |
| alligator | arm (96.3%) | Arm extension similar |

## Confusion Matrix Analysis

I generated a confusion matrix on the test set which revealed:

**Key Finding:** "another" acts as a dominant attractor many signs get misclassified as "another". This suggests the model learned overly broad patterns for certain signs.

**Per-sign accuracy on test data:**
- Best: apple (91%), another (85%), any (85%), all (83%)
- Worst: airplane (51%), after (68%), alligator (70%)

**Important Insight:** Signs with similar hand positions or movements confuse the model, regardless of how many training videos they have. For example:
- "arm" works at 99.2% with only 300 videos (fewest in dataset)
- "another" fails despite having 557 videos (most in dataset)

This disproves my initial hypothesis that more videos = better accuracy. **Sign distinctiveness matters more than quantity.**

## Timing Discovery

I discovered that signing speed affects recognition:
- Some signs work even when performed quickly
- Other signs only work when performed slowly with a pause at the end
- This suggests the model learned temporal patterns specific to PopSign's recording conditions

## Updated Metrics Summary

| Metric | Value |
|--------|-------|
| Test Accuracy (held-out data) | 72.8% |
| Real-World Accuracy (on me) | 54% (7/13 signs) |
| High-Confidence Signs | 3 (arm, hello, apple) |
| Performance Gap | ~19% |

## New Direction: Sign-to-Sentence System

Based on these findings, I am pivoting my approach. Rather than trying to recognise all signs with low accuracy, I will:

**Focus on a practical communication system:**

1. **Keep only reliably-working signs** - Select 7-10 signs that achieve >80% real-world accuracy
2. **Add sentence generation** - Use NLP to convert recognised sign sequences into natural sentences

**Example workflow:**
```
User signs: [hello] → [chair] → [sit]
System predicts: ["hello", "chair", "sit"]
NLP generates: "Hello, I would like to sit on the chair"
```

**Why this approach:**
- More practical for real communication
- Doesn't require signs for grammar words (I, want, to, the, etc.)
- Demonstrates end-to-end system thinking
- Works around vocabulary limitations intelligently

## Key Learnings for Dissertation

1. **Test accuracy ≠ Real-world accuracy** - The 72.8% → 54% gap is significant and worth discussing
2. **Sign distinctiveness > Data quantity** - Similar signs confuse the model regardless of training samples
3. **Domain adaptation is crucial** - Model learned PopSign-specific patterns (signers, camera, timing)
4. **Practical systems need pragmatic solutions** - Focusing on working signs + NLP may be more valuable than perfect recognition

## Next Steps

1. Select final set of distinct, working signs (7-10 signs) and gradually add more vocabulary
2. Potentially process more signs from PopSign to find better candidates (e.g., eat, drink, help, please, thank you)
3. Implement sign collection feature (record multiple signs in sequence)
4. Integrate NLP for sentence generation
5. Build demo showcasing the complete pipeline

## Reflection

This debugging process, while time consuming, has been valuable. I now understand that the challenge isn't just "getting more data" or "fixing bugs" it's the fundamental difficulty of generalising sign language recognition to new signers and environments. This insight will strengthen my dissertation's discussion of real-world deployment challenges.

The pivot to a sign-to-sentence system demonstrates practical problem solving: rather than being blocked by recognition limitations, I'm designing around them to create something genuinely useful.

