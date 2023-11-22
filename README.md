<h2>A/B Hypothesis Testing: Ad campaign performance </h2>

This repository contains code to solving an Ad-Campaign business problem, using A/B testing.

The main objective of this project is to test if the ads that an advertising company ran resulted in a significant lift in brand awareness.

Check out the deployed app [here](https://share.streamlit.io/katenjoki/a-b-hypothesis-testing-ad-campaign-performance/scripts/app.py#a-b-hypothesis-testing-ad-campaign-performance)!

<h3>Data</h3>
The data for this project is a “Yes” and “No” response of online users to the following question:

**Q: Do you know the brand Lux?**
O  Yes		O  No
		
The users that were presented with the questionnaire above were chosen according to the following rule:

* Control: users who have been shown a dummy ad
* Exposed: users who have been shown a creative that was designed by SmartAd for the client. 


<h3> Simple A/B testing </h3>
This involves comparing two versions of a variable/ product to see which perfroms better, in a controlled experiment.

**Null Hypothesis:-** The creative ad designed by SmartAd did not result in a significant lift in brand awareness (when compared to the dummy ad).

Number of users per group (control, exposed) who responded yes (1) or no (0) to the question: Do you know the brand Lux?

| Response | 0 | 1 |
| ---------| ---------| ---------|
| experiment |  |  |
| control | 322  | 264  |
| exposed | 349 | 308 |

**Basic stats**
|  | response_rate | std deviation | std_error |
| ---------| ---------| ---------| ----------|
| experiment |  |  |  | 
| control | 0.451 | 0.498  | 0.020  |
| exposed | 0.469 | 0.499 | 0.019 |

**Interpreation:**
* Around 45.1% of users in the control group and 46.9% of users in the exposed group responded positively to the question.
* The exposed group has a slightly higher response rate, but we have to determine whether the difference is *statistically significant.* 

**Statistical significance of A/B test**
A z test is used to test a null hypothesis by comparing the means of 2 groups and is used when the sample size > 30.
* z_statistic: -0.65
* p value: 0.518
* Confidence interval(95%) for control group: [0.410,0.491]
* Confidence interval(95%) for exposed group: [0.431,0.507]

**Conclusion**
Using a significance level of 0.05, we can observe that the p value of 0.518 is much greater. This means we *fail to reject the null hypothesis*. A low p-value prevents us from making a Type I error which occurs when we reject the null hypothesis when it's true in the population, leading to a false positive. 

We conclude that the creative ad designed by SmartAd did not result in a significant lift in brand awareness. 

