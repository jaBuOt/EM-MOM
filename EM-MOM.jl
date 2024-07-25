using Pkg 
using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter
using JLD2
using Distributions
using Parameters
using StatsBase
using CSV
using DataFrames
using MacroTools
using Base.Threads
using StaticArrays

# Performs forward propagation where the ro, a and filter functions
# are updated. Returns the updated ro, a and filter

function forwardProp(aa, chi, chi0, fil, mu, p, ro)

    aa[1] = 0

    for i ∈ 1:xLen

        ro[1, i] = 0
        
        for j ∈ 1:xLen
            for k ∈ 1:yLen
                # print("hello")
                # print(findfirst(isequal(Y[1]), ySpace))
                ro[1, i] = ro[1, i] .+ mu[j, k] * p[j, k, i, findfirst(isequal(Y[1]), ySpace)]
                # print(ro[1,i])
            end
        end
        aa[1] += ro[1, i]
    end

    # print("aa1 ", string(aa[1]))

    for i ∈ 1:xLen     
        # allowed? 
        if aa[1] == 0
            fil[1, i] = 0
        else  
            fil[1, i] = ro[1, i] / aa[1]
        end
        # print("fil1i ", string(fil[1, i]))
    end
            
    for i ∈ 2:N

        aa[i] = 0

        for j ∈ 1:xLen

            ro[i, j] = 0

            for k ∈ 1:xLen
                ro[i, j] = ro[i, j] .+ fil[i-1, k] * p[k, findfirst(isequal(Y[i-1]), ySpace), j, findfirst(isequal(Y[i]), ySpace)]
            end
            aa[i] += ro[i, j]
        end

        for j ∈ 1:xLen
            # allowed? 
            if aa[i] == 0
                fil[i, j] = 0
            else
                fil[i, j] = ro[i, j] / aa[i]
            end
        end
    end

    aa, chi, chi0, fil, mu, p, ro
end

# Performs backward propagation where the chi and chi0 functions
# are updated. Returns the updated chi and chi0

function backwardProp(aa, chi, chi0, fil, mu, p, ro)

    for i ∈ 1:xLen
        for j ∈ 1:xLen

            den = 0
            for k ∈ 1:yLen
                den = den .+ p[i, findfirst(isequal(Y[N-1]), ySpace), j, k]
            end

            if den == 0
                chi[N-1, i, j] = 0
            else
                chi[N-1, i, j] = p[i, findfirst(isequal(Y[N-1]), ySpace), j, findfirst(isequal(Y[N]), ySpace)] ./ den
            end
            # print("chi",  string(chi[N-1, i, j]))
        end
    end

    for i ∈ 2:N-1
        for j ∈ 1:xLen
            for k ∈ 1:xLen

                den = 0
                for l ∈ 1:yLen
                    den = den .+ p[j, findfirst(isequal(Y[N-i]), ySpace), k, l]
                end

                sum = 0
                for l ∈ 1:xLen
                    mul = 0
                    for m ∈ 1:yLen
                        mul = mul .+ p[k, findfirst(isequal(Y[N-i+1]), ySpace), l, m]
                    end
                    sum = sum .+ chi[N-i+1, k, l] * mul
                end

                # allowed?
                if aa[N-i+1] * den == 0
                    chi[N-i, j, k] = 0
                else
                    chi[N-i, j, k] = p[j, findfirst(isequal(Y[N-i]), ySpace), k, findfirst(isequal(Y[N-i+1]), ySpace)] * sum ./ (aa[N-i+1] * den)
                end
            end
        end
    end

    for i ∈ 1:xLen
        for j ∈ 1:xLen
            for k ∈ 1:yLen

                den = 0
                for l ∈ 1:yLen
                    den += p[i, k, j, l]
                end

                # print(den, " ")

                sum = 0
                for l ∈ 1:xLen
                    mul = 0
                    for m ∈ 1:yLen
                        mul = mul .+ p[j, findfirst(isequal(Y[1]), ySpace), l, m]
                    end
                    sum = sum .+ chi[1, j, l] * mul
                end

                # is this allowed?
                if aa[1] * den == 0
                    chi0[i, j, k] = 0
                else
                    chi0[i, j, k] = p[i, k, j, findfirst(isequal(Y[1]), ySpace)] * sum ./ (aa[1] * den)
                end
            end
        end
    end

    return aa, chi, chi0, fil, mu, p, ro
end

# Updates and returns the current transition probability function p

function updateP(aa, chi, chi0, fil, mu, p, ro)

    for i ∈ 1:xLen
        for j ∈ 1:yLen
            for k ∈ 1:xLen
                for l ∈ 1:yLen

                    sum = 0

                    for m ∈ 1:N-1
                        # print(findfirst(isequal(Y[m]), ySpace))
                        # if indexin(Y[m], ySpace) == j && indexin(Y[m+1], ySpace) == l
                        if findfirst(isequal(Y[m]), ySpace) == j && findfirst(isequal(Y[m+1]), ySpace) == l
                            # print("hello")
                            sum += chi[m, i, k] * fil[m, i]
                        end
                    end

                    # print(sum, " ")

                    prob = 0
                    
                    for m ∈ 1:yLen
                        prob += p[i, j, k, m]
                    end

                    # print(prob)

                    if findfirst(isequal(Y[1]), ySpace) == l
                        # print("hello")
                        num = prob * (chi0[i, k, j] * mu[i, j] + sum)
                    else
                        num = prob * sum
                    end
                    # print(num, " ")

                    den = 0 

                    for m ∈ 1:xLen

                        sum = 0

                        for n ∈ 1:N-1
                            # if indexin(Y[n], ySpace) == j 
                            if findfirst(isequal(Y[n]), ySpace) == j
                                sum += chi[n, i, m] * fil[n, i]
                            end
                        end

                        prob = 0

                        for n ∈ 1:yLen
                            prob += p[i, j, m, n]
                        end

                        den = den .+ prob * (chi0[i, m, j] * mu[i, j] + sum)
                        # print(den)
                    end

                    # allowed?
                    if den == 0
                        p[i, j, k, l] = 0
                    else
                        p[i, j, k, l] = num / den
                    end
                    # print(p[i, j, k, l])
                    # print("p ", string(p[i,j,k,l]))
                end
            end
        end
    end

    return aa, chi, chi0, fil, mu, p, ro
end

# Updates and returns the current initial probability density mu

function updateMu(aa, chi, chi0, fil, mu, p, ro)

    for i ∈ 1:xLen
        for j ∈ 1:yLen

            num = 0

            for k ∈ 1:xLen
    
                prob = 0

                for l ∈ 1:yLen
                    prob += p[i, j, k, l]
                end

                num += chi0[i, k, j] * prob
            end

            num = mu[i, j] * num

            den = 0

            for k ∈ 1:xLen
                for l ∈ 1:yLen

                    sum = 0

                    for m ∈ 1:xLen

                        prob = 0

                        for n ∈ 1:yLen
                            prob += p[k, l, m, n]
                            # print(p[k, l, m, n])
                        end

                        # print(prob)

                        sum += chi0[k, m, l] * prob
                    end

                    # print(sum)

                    den += mu[k, l] * sum
                end
            end

            # allowed?
            if den == 0
                mu[i, j] = 0
            else
                mu[i, j] = num / den
            end

            # print(den)
        end
    end

    return aa, chi, chi0, fil, mu, p, ro
end

# Calculate difference between current p and mu and previous p and mu 
# to assess convergence. Returns a cumulative difference for p and mu

function calcConvergence(pCopy, muCopy, p, mu)

    pDiff = 0
    muDiff = 0

    for i ∈ 1:xLen
        for j ∈ 1:yLen
            for k ∈ 1:xLen
                for l ∈ 1:yLen
                    pDiff += abs(p[i, j, k, l] - pCopy[i, j, k, l])
                end
            end
            muDiff += abs(mu[i, j] - muCopy[i, j])
        end
    end

    return pDiff, muDiff
end


function emMom()

    # Initialize p

    p = @MArray zeros(Float64, xLen, yLen, xLen, yLen)

    for i1 ∈ 1:xLen
        for i2 ∈ 1:yLen
            for i3 ∈ 1:xLen
                for i4 ∈ 1:yLen

                    p[i1, i2, i3, i4] = 1 / (xLen * yLen)
                end
            end
        end
    end

    # Initialize mu

    mu = @MArray zeros(Float64, xLen, yLen)

    for i1 ∈ 1:xLen
        for i2 ∈ 1:yLen
            mu[i1, i2] = 1 / (xLen * yLen)
        end
    end

    chi = @MArray zeros(Float64, N-1, xLen, xLen)
    chi0 = @MArray zeros(Float64, xLen, xLen, yLen)
    ro = @MArray zeros(Float64, N, xLen)
    fil = @MArray zeros(Float64, N, xLen)
    aa = @MArray zeros(Float64, N)

    pDiff = 10
    muDiff = 10

    while (pDiff > 0.05) || (muDiff > 0.05)
    #Threads.@threads for i ∈ 1:1000000

    #    if (pDiff <= 0.05) && (muDiff <= 0.05)
    #        break
    #    end

        pCopy = copy(p)
        muCopy = copy(mu)
        
        # Forward propagation
        aa, chi, chi0, fil, mu, p, ro = forwardProp(aa, chi, chi0, fil, mu, p, ro)

        # Backward propagation
        aa, chi, chi0, fil, mu, p, ro = backwardProp(aa, chi, chi0, fil, mu, p, ro)

        # Probability update
        aa, chi, chi0, fil, mu, p, ro = updateP(aa, chi, chi0, fil, mu, p, ro)
        aa, chi, chi0, fil, mu, p, ro = updateMu(aa, chi, chi0, fil, mu, p, ro)

        # Calculate difference between current p and mu and previous p and mu 
        # to assess convergence
        pDiff, muDiff = calcConvergence(pCopy, muCopy, p, mu)
    end

    return p, mu
end

# Assigns a bin to each observation and creates the new array of observations

function binData(data, binSize)

    obs = []

    for i ∈ 1:length(data)
        # obs[i] = data[i] ÷ binSize
        push!(obs, trunc(Int, (round(data[i] / binSize))) * binSize)
    end

    return obs
end

function getYSpace(data, binSize)

    min = data[1]
    max = data[1]
    
    for i ∈ 1:length(data)
        if data[i] > max
            max = data[i]
        elseif data[i] < min
            min = data[i]
        end
    end

    ySpace = []

    for i ∈ min:binSize:max
        push!(ySpace, i)
    end

    return ySpace

end

# data = CSV.read("C:/Users/Jaime/OneDrive/Desktop/LCD_sample.csv", DataFrame, select=["HourlyDryBulbTemperatureC"]) # open and read data file
data = CSV.read("C:/Users/Jaime/OneDrive/Desktop/Summer Kouritzin Research/weatherData/PADUCAH BARKLEY, KY, 2022.csv", DataFrame, select=["DailyAverageDryBulbTemperature"]) # open and read data file
data = dropmissing(data, disallowmissing=true) # disregard missing data values

unbinnedData = Array(data)
unbinnedData = unbinnedData[1:10]
binWidth = 1

Y = binData(unbinnedData, binWidth) # bin data

print(Y)

ySpace = getYSpace(Y, binWidth) # get Y state space

print(ySpace)

xSpace = ySpace # get X state space

#=
xSpace = [-30+n/10 for n=0:500] # dew point temperature (range is -30 to 20 by 0.1 increments)
ySpace = [-20+n/10 for n=0:400] # hourly dry bulb temperature in Celsius (range is -20 to 20 by 0.1 increments))
=#

# These are sample observations and X and Y state spaces for testing the EM-MOM algorithm

###################################################################

# Y = [4, 2, 4, 5, 6, 5, 6, 7, 5, 3, 3, 3, 5, 4, 5, 5, 4, 6, 7, 7]
# xSpace = [0.5, 1.25, 2, 3.1, 4.4, 5, 5.8, 7.2, 8]
# ySpace = [1, 2, 3, 4, 5, 6, 7]

# Y = [4, 2, 4]
# xSpace = [0.5, 1.25]
# ySpace = [2, 3, 4]

###################################################################

N = length(Y)
xLen = length(xSpace) # size of hidden state space
yLen = length(ySpace) # size of observable state space
#=
Y = [1, 2, 3]
N = 3
xSpace = [1, 2, 3]
ySpace = [1, 2, 3]
xLen = 3
yLen = 3
=#

# Alternative way of initializing the transition probabilities p. 
# This approach follows the initialization approach in the Markov 
# Observation Models paper

###################################################################
#=
transitions = zeros(Float32, yLen, yLen)

for i1 ∈ 1:yLen
    for i2 ∈ 1:yLen
        for i ∈ 1:N-1

            if Y[i] == ySpace[i1] && Y[i+1] == ySpace[i2]

                transitions[i1, i2] += 1
            end
        end
    end
end 

p = zeros(Float64, xLen, yLen, xLen, yLen)

for i1 ∈ 1:xLen
    for i2 ∈ 1:yLen
        for i3 ∈ 1:xLen
            for i4 ∈ 1:yLen

                if sum(transitions[i2,:]) > 0
                    p[i1, i2, i3, i4] = transitions[i2, i4] / (sum(transitions[i2,:]) * xLen)
                else
                    p[i1, i2, i3, i4] = 0
                end
            end
        end
    end
end
=#
###################################################################

pFinal, muFinal = @time emMom() # run EM-MOM algorithm to get final probability estimates

print(pFinal, "\n")
print(muFinal, "\n")








