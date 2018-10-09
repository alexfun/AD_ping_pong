// taken from  http://tobiasmadsen.com/2017/01/09/automatic_differentiation_in_r

// [[Rcpp::depends(rstan)]]


#include <Rcpp.h>
#include <stan/math.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector stan_obj_fun(NumericVector par,
                           IntegerVector me,
                           IntegerVector you,
                           IntegerVector score_diff) {

    
    // parameters:
    stan::math::var p_0 = par[0];
    stan::math::var alpha_0 = par[1];
    stan::math::var alpha_1 = par[2];
    stan::math::var beta = par[3];
    
    int n_obs = me.size();

    NumericVector ret(5); // initialise return object for obj fn and gradient
    stan::math::var p = p_0; // initialise initial model fitted probability
    stan::math::var nll = 0; // initialise log-lik
    
    for (int i = 0; i < n_obs; i++) {

        if (i > 0) {
            // mixing coefficient
            stan::math::var alpha = inv_logit(alpha_0 + alpha_1 * score_diff[i - 1]);
            // update p with model
            p =  alpha * p_0 + (1 - alpha) *  inv_logit(logit(p) + beta);
        }
        {
        // calculate probability of observing score-line, and take its log
        // this is essentially the function get_prob_of_scoreline met in other examples
        IntegerVector tmp_vec(2); // it's annoying, but we need to do this to get pairwise minimum
        tmp_vec[0] = me[i];
        tmp_vec[1] = you[i];
        int min_me_you = min(tmp_vec);
        if (you[i] >= 20) {
            nll += \
                -(stan::math::binomial_coefficient_log(40, 20) + 
                log(pow(2, min_me_you - 20) * pow(p, me[i]) * pow(1 - p, you[i])));

        } else {
            nll += \
                -(stan::math::binomial_coefficient_log(20 + min_me_you, min_me_you) + 
                log(pow(p, me[i]) * pow(1 - p, you[i])));

        }
        
        }


    }
    
    // Compute Function Value
    ret[0] = nll.val();

    // Compute Gradient
    nll.grad();
    ret[1] = p_0.adj();
    ret[2] = alpha_0.adj();
    ret[3] = alpha_1.adj();
    ret[4] = beta.adj();
    
    // Memory is allocated on a global stack
    stan::math::recover_memory();
    stan::math::ChainableStack::memalloc_.free_all();
    
    return ret;
}
