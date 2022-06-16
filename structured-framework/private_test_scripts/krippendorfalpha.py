#! /usr/bin/env python
# -*- coding: utf-8
'''
Python implementation of Krippendorff's alpha -- inter-rater reliability
(c)2011-17 Thomas Grill (http://grrrr.org)
Python version >= 2.4 required
'''

from __future__ import print_function
try:
    import numpy as np
except ImportError:
    np = None


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=str, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))


    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
    
    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)
    
    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.


if __name__ == '__main__': 
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    data = (
        #"B B A C A C A A D A A B B B A A C A C D", # coder A
        #"B B A C A C B A D A A C B B A A C A C D", # coder B
        #"B B A C A C A A D A A B B B A D C A C D", # coder C
        #"C B A C A C A A D B A A B B A D A A C D",
        #"C B B C A A A A C B A B A A A D C A A D",
        #"C B B C A C A A D C A A A B A D B A C C"
        #"C B A C A B D B D D A B A B A D C A C C",
        #"D A A C A A A B D C A D B B A D C D B D",
        #"C B A C C C C B D B A B A B A D B A D D",
        #"C B A B B A A A D B A B A B A D A A C D",
        #"C D B C B C A A D B A B B B A D B A A D",
        #"C B B C B A A A D B A C A B A D C C C D"
        #"C B A C A C A A D A A B B B A D C A D D",
        #"C B A C A C A A D B A A B B A A C A D D",
        #"C B A B A B A B D C A A B B A D C B B D"
        #"A A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11",
        #"A A1 A2 A3 B4 A5 A6 B7 A8 B9 B10 B11",
        #"A A1 A2 A3 C4 B5 A6 C7 A8 B9 A10 B11"

        #"A A1 A2 A3 A4 A5 A6 A7 B8 B9 A10 A11",
        #"B A1 A2 A3 A4 B5 B6 B7 B8 B9 A10 B11",
        #"B A1 B2 B3 A4 C5 A6 B7 B8 B9 A10 C11"
        #"A A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11",
        #"B B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11",
        #"C A1 A2 A3 A4 A5 C6 A7 A8 C9 A10 C11"
        #"T S T Dr H We T H S Co Dr We Dr Do Dr Do Wa H Do Do",
        #"T S Cr S T Wa A Cr S Co A We Dr Do Dr Wa Wa S S Do",
        #"Cr S Cr Co T T Dr H A A A We Do Do Dr Wa Wa S S Do"
        #"Cr T Cr S S We Cr H A Cr Co We Do We Dr Do Wa H Do Do",
        #"A S Cr Co S We A H T Co A We Dr Do Co Do Wa H S Co",
        #"Cr S Cr Cr Cr Cr Cr H Wa Cr Dr Dr Dr Do Dr Do Wa S S Co"
        #"Cr S Cr Cr Cr We Cr Cr S Co A We Dr Dr Dr Wa Wa T S Co",
        #"Cr A Cr Dr Cr We A T Wa We A We Dr Do Dr Wa A Dr A Co",
        #"Cr S Cr Dr Cr We Cr H Wa We A We Dr Do Do Wa Wa S S Do"
        #"Cr S Cr Dr Cr We Cr H A We T T Do Do Do Wa Wa S S Dr",
        #"We S Cr T Cr We Cr H A We T We Do Do Dr Wa Wa S S Do",
        #"A Dr Cr Cr Cr We A H Do We T We A A T Wa Wa S S Co"
        #"A A B A B B A A A B",
        #"A A A A B B A A A B"
        #"B B A A C B A B B A B C A A A B A C B C C C C",
        #"B B A A B B A B B A B C A A A B A A B C C C C"
        #"p n n p p p p n p n p p n n n p n p p p",
        #"p p n n p p p n p p n p p n p p n p p n",
        #"p n p p p p p n p n n p p n p p n p p p",
        #"A A",
        #"A A"
        #"p n n p p p n n p n p n p n p p n p p p",
        #"p p n n p p n n p p p n p n p n p p p n",
        #"p p n n p p n n p p p p p p n n n p p n"
        #"p n n p p p n p p n p n p n p p n p p p",
        #"p n n p p p n p p n p p p n p p n p p p",
        #"p p p p p p n p p p p p p p p p p p p p"
        #"B B B B B B B B C C",
        #"B C B B B C B B B C"
        "A B A A C C A A C A A C B B B A C A A C C B C",
        "B C C A C C A A C A B C A A B B C A A C C A C"
    )

    missing = '*' # indicator for missing values
    array = [d.split(' ') for d in data]  # convert to 2D list of string items
    
    print("nominal metric: %.3f" % krippendorff_alpha(array, nominal_metric, missing_items=missing))
