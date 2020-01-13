#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:10:45 2020

@author: shaheer
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import requests
res = requests.post('http://7ff078b7.ngrok.io/similarity/123', 
                    json={
                            "blogs":
                                ['''Merge sort is a sorting technique based on divide and conquer technique. With worst-case time complexity being Ο(n log n), it is one of the most respected algorithms.

Merge sort first divides the array into equal halves and then combines them in a sorted manner.
How Merge Sort Works?

To understand merge sort, we take an unsorted array as the following −
Unsorted Array

We know that merge sort first divides the whole array iteratively into equal halves unless the atomic values are achieved. We see here that an array of 8 items is divided into two arrays of size 4.
Merge Sort Division

This does not change the sequence of appearance of items in the original. Now we divide these two arrays into halves.
Merge Sort Division

We further divide these arrays and we achieve atomic value which can no more be divided.
Merge Sort Division

Now, we combine them in exactly the same manner as they were broken down. Please note the color codes given to these lists.

We first compare the element for each list and then combine them into another list in a sorted manner. We see that 14 and 33 are in sorted positions. We compare 27 and 10 and in the target list of 2 values we put 10 first, followed by 27. We change the order of 19 and 35 whereas 42 and 44 are placed sequentially.
Merge Sort Combine

In the next iteration of the combining phase, we compare lists of two data values, and merge them into a list of found data values placing all in a sorted order.
Merge Sort Combine

After the final merging, the list should look like this −
Merge Sort

Now we should learn some programming aspects of merge sorting.
Algorithm

Merge sort keeps on dividing the list into equal halves until it can no more be divided. By definition, if it is only one element in the list, it is sorted. Then, merge sort combines the smaller sorted lists keeping the new list sorted too.''',
'''Fusion classification is a classification technique based on the technique of dividing and conquering. In the worst case, being the complexity of time Ο (n log n), it is one of the most respected algorithms.

Merge sorting first divides the table into equal halves and then combines them ordered.
How does fusion classification work?

To understand the type of merger, let's take an unordered matrix as follows:
Unclassified table

We know that fusion ordering first divides the entire matrix iteratively into equal halves, unless atomic values ​​are reached. We see here that an array of 8 elements is divided into two matrices of size 4.
Merge the type

This does not change the sequence of appearance of the elements in the original. Now we divide these two tables into two.
Merge the type

Then we divide these tables and obtain an atomic value that can no longer be divided.
Merge the type

Now we combine them in exactly the same way they were broken down. Consider the color codes given to these lists.

First we compare the element of each list, then we combine them ordered in another list. We see that 14 and 33 are in orderly positions. We compare 27 and 10 and in the objective list of 2 values ​​we put 10 first, followed by 27. We change the order of 19 and 35, while 42 and 44 are placed sequentially.
Merge the type

In the next iteration of the combination phase, we compare the lists of two data values ​​and merge them into a list of found data values, putting everything in order.
Merge the type

After the final merger, the list should look like this:
Sort Merge

Now we should learn some aspects of fusion type programming.
Algorithm

The type of merger continues to divide the list into equal halves until it can no longer be divided. By definition, if it is only one item in the list, it is ordered. Then, the combined sort combines the smaller ordered lists, while maintaining the new list.''','''Bush are an English rock band formed in London, England in 1992. Their current lineup consists of lead vocalist and rhythm guitarist Gavin Rossdale, drummer Robin Goodridge, lead guitarist Chris Traynor, and bassist Corey Britz.

In 1994, Bush found immediate success with the release of their debut album, Sixteen Stone, which is certified 6× multi-platinum by the RIAA.[3] They went on to become one of the most commercially successful rock bands of the 1990s, selling over 10 million records in the United States and 20 million in the world.

Despite their success in the US (especially in the mid-1990s), the band were considerably less popular in their home country – a period when Britpop groups dominated the UK charts and the appeal of the grunge sound had declined – and they have enjoyed only marginal success there.[4]

Bush have had numerous top ten singles on the Billboard rock charts[5] and one No. 1 album with Razorblade Suitcase in 1996.[6] The band broke up in 2002 but reformed in 2010, and have released three albums since then: The Sea of Memories (2011), Man on the Run (2014), and Black and White Rainbows (2017).[7] In late 1996 Bush released the first single "Swallowed" from their second album titled Razorblade Suitcase. The song spent seven weeks on top of the Modern Rock Tracks chart. This was followed by single "Greedy Fly". The album hit number 1 in America and placed high in many European countries. Razorblade Suitcase featured American recording engineer Steve Albini, a move which was viewed negatively by critics. Albini had worked with Nirvana on their final studio album, In Utero, three years before. Bush later released the remix album Deconstructed. The album saw Bush re-arranging their songs into dance and techno stylings. The album went platinum less than a year after release.
The Science of Things (1999–2000)''','''By October, with the IPO abandoned and his office-space sharing company bleeding cash, Neumann found himself late on a Sunday evening pleading with WeWork's largest lender for a $5 billion lifeline, people familiar with the matter said.

The message was clear — without new financing WeWork would run out of money within weeks.

“Do you still believe in the company?” Neumann, who had stepped down as CEO but was still WeWork's chairman, asked a room of JPMorgan Chase and Co bankers on the 42nd floor of their midtown Manhattan headquarters on October 6, the sources said.The JPMorgan bankers, led by asset and wealth management CEO Mary Erdoes and debt capital markets head Jim Casey, told Neumann and other WeWork directors they would back the company, and were confident they could raise the money. But they would not underwrite the deal on the spot, as one board member requested.

With questions swirling around WeWork's chances of survival in the wake of its failed IPO, the bankers told Neumann they needed some time to sound out investors first, according to the sources.

A few days later, an alternative rescue plan began to emerge from WeWork's largest shareholder, Japan's SoftBank Group Corp .

Both were far from perfect. But given its dire straits, WeWork was fortunate to have a choice.

This account of how WeWork's financial rescue came together over the past three weeks is based on interviews with eight people with knowledge of the negotiations. They requested anonymity to discuss the confidential deliberations.

WeWork, SoftBank and JPMorgan declined to comment for this story. Requests for an interview with Neumann were also declined.
Corporate governance problem

SoftBank offered $9.5 billion to WeWork, including new debt and recommitted equity, as well as a tender offer to partly cash out Neumann and other shareholders.

In addition to providing more funds than JPMorgan, the SoftBank deal resolved what some WeWork directors privately referred to as the company's “corporate governance problem” — Neumann's controlling grip.

SoftBank's deal would strip Neumann's voting power and remove him from the board. Neumann was blamed by other WeWork investors such as Benchmark Capital and China's Hony Capital, for the company's precipitous decline, some of the people said.

His erratic management style, combined with WeWork's lack of a clear path to profitability, alienated potential IPO investors.

The problem was that Neumann could still wield power over the company even after he quit as CEO on September 24 because, as WeWork's founder, each of his shares had 10 voting rights. Others had only one vote for every share.

It was clear to SoftBank, as well as to a special board committee formed to consider the financing plans, that Neumann relinquishing control would come with a price tag, three of the people said.
Claure and Neumann negotiate

In meetings in New York between Neumann and SoftBank Chief Operating Officer Marcelo Claure, the contours of a side deal came together.

SoftBank would provide a $500 million credit line to refinance Neumann's personal borrowings made against WeWork's stock, as long as he used proceeds from cashing out up to $970 million of his shares to repay SoftBank for that loan first.

SoftBank's latest offer valued the company at as little as $5.9bn based on the repricing of warrants it was already committed to exercise, according to Bernstein research, a far cry from the $47bn it had assigned to WeWork in January.

Neumann's special pay-off did not stop there. He negotiated with SoftBank a four-year non-compete agreement with a $185 million “consulting fee” in return for stepping down from WeWork's board. He would now only get to observe board proceedings instead of participating.

WeWork's special board committee members expressed concern that Neumann's bailout would cause outrage among many WeWork employees whose stock options had a much higher strike price than the valuation in SoftBank's tender offer, some of the people said.

The committee declined to comment for this article.

SoftBank's offer to buy up to $3bn of WeWork stock from employees and existing shareholders would value the company at about $8bn, higher than the new valuation based on the warrants.

Neumann attempted to push up the valuation of the tender offer in his negotiations with SoftBank, one of the people said.

SoftBank stood firm. Anticipating criticism from its own shareholders for possibly throwing good money after bad, it wasn't prepared to pay more.

Most WeWork directors wanted Neumann off the board, and many minority shareholders, which the special committee was formed to represent, wanted to cash out, the people said.

JPMorgan had not pursued raising additional money for shareholders, in part because WeWork only tasked the bank with delivering $5bn of debt financing.
JPMorgan's covenant

When the bank submitted the debt package on Monday, only private equity firm Starwood Capital Group, run by real estate mogul Barry Sternlicht, had committed to join JPMorgan in sharing the financing burden

JPMorgan agreed to provide the rest and transfer the money by Thursday. But the deal also included a condition, known as a covenant, that would trigger a debt default if SoftBank did not make good on $1.5bn that it had already committed to provide when warrants came due next April, the people said.

JPMorgan wanted to ensure SoftBank would honor its financial commitment, which would lessen the risk for investors holding debt that would one day need to be repaid.

But SoftBank told WeWork's board that it would not pay the $1.5bn if its financing offer was rejected, two of the people added.

The problem from SoftBank's perspective was that those warrants were based on January's $47bn valuation of WeWork. It wanted to change the pricing radically to reflect WeWork's dramatic fall in value.

JPMorgan deal makers, on the other hand, believed WeWork's contract with SoftBank prevented the telecommunications and technology giant from reneging on that earlier commitment, other people familiar with the matter said.

In the end, the risk that SoftBank wouldn't fulfill its commitment if the JPMorgan financing package was chosen helped swing the special committee's deliberations in favor of SoftBank's offer, they added.

SoftBank had made WeWork an offer it had to accept.

On Tuesday, SoftBank unveiled a deal that increased its ownership of WeWork to 80 per cent from 30pc but sought to avoid having to consolidate WeWork's liabilities on its balance sheet by not taking full control of the expanded board. SoftBank will only have 5 of the 10 seats.

But unless SoftBank can turn WeWork around quickly, it could be a pyrrhic victory.

WeWork has burned through almost $2.5bn of cash since the end of June, and Claure will have to cut costs quickly, including slashing thousands of jobs and helping the company to find a way to get out of expensive leases. Otherwise, the new cash injection may not last long enough to stabilise the business.

As predicted, Claure is also facing some staff anger over the size of Neumann's payout. He told employees on Wednesday that it was the “price” that had to be paid to get rid of Neumann's voting rights. Otherwise, he said, “Adam could do whatever he wanted.”''']
                                })

if res.ok:
    print(res.json())
else:
    print("somthing went wrong")
    
def similarity(docs,document):
    data = []
    sim = cosine_similarity(docs,document)
    sim = sim.flatten()
    sim = np.array(sim)
    #sim = sim.reshape(-1,1)
    #sim = np.sort(sim)
    for i in range(len(sim)):
        data.append((i,sim[i]))
    sim = sim[~(sim >= 1.0)]
    data.sort(key = lambda x: x[1])
    return data

def getVectors(corpus,vectors,vocab_size):
    wordset = set(vectors.wv.index2word) #Checks if the word is in the Word2vec corpus 
    counter = 0    
    featureVec = np.zeros(vocab_size,dtype="object")
    for word in corpus:
      if word in wordset:
        featureVec = np.add(featureVec,vectors[word])
      counter = counter + 1
      #print(counter)
    featureVec = np.divide(featureVec,counter)
    return featureVec

docs = []

doc1 = "hello world"
doc2 = "hello babu"
doc3 = "hello saeen"
doc1 = getVectors(doc1,vectors,300)
doc2 = getVectors(doc2,vectors,300)
doc3 = getVectors(doc3,vectors,300)
#doc1 = doc1.reshape(1,-1)
docs.append(doc1)
docs.append(doc2)
docs.append(doc3)

docs = np.array(docs)
doc1 = doc1.reshape(1,-1)
#docs = docs.reshape(1,-1)

data = similarity(docs,doc1)

print(data[-1])


