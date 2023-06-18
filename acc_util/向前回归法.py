import statsmodels.formula.api as smf

def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(response, ' + '.join(selected + [candidate]))
            aic = smf.logit(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_candidates.pop()

        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('aic is {},continuing!'.format(current_score))

        else:
            print('forward selection over!')
            break
    
    formula = "{} ~ {} ".format(response, ' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.logit(formula=formula, data=data).fit()
    return(model)