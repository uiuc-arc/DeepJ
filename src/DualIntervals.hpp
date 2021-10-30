#pragma once
#include <math.h>

#include <boost/logic/tribool.hpp>
#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/io.hpp>
#include <boost/numeric/interval/rounded_arith.hpp>
#include <cmath>
#include <iostream>
#include <utility>

static const double ulp = ldexpl(1.0, -52);
static const double min_denormal = ldexpl(1.0, -1074);

namespace bn = boost::numeric;
namespace bni = bn::interval_lib;
typedef bni::checking_no_nan<double> checking;
typedef bn::interval<double, bni::policies<bni::save_state<bni::rounded_transc_std<double>>, checking>> Interval;

////////////////////////////////////////////////////////////////////////////////
//
// DUAL NUMBER INTERVAL CLASS
//
////////////////////////////////////////////////////////////////////////////////

class DualInterval {
   public:
    Interval real;
    Interval dual;

    DualInterval(double x1, double x2, double x3, double x4) {
        this->real = Interval(x1, x2);
        this->dual = Interval(x3, x4);
    }

    DualInterval(const Interval& i1, const Interval& i2) {
        this->real = i1;
        this->dual = i2;
    }

    DualInterval(double x1) {
        this->real = Interval(x1, x1);
        this->dual = Interval(0, 0);  // if no dual part is provided, set it to 0
    }

    DualInterval(int x1) {
        this->real = Interval(double(x1), double(x1));
        this->dual = Interval(0, 0);  // if no dual part is provided, set it to 0
    }

    DualInterval() {  // if neither a real or dual part is given everything initialized to 0
        this->real = Interval(0, 0);
        this->dual = Interval(0, 0);
    }

    DualInterval operator+(const DualInterval& rhs) const;
    DualInterval operator+(const double rhs) const;
    friend DualInterval operator+(const double lhs, const DualInterval& rhs);

    DualInterval operator-(const DualInterval& rhs) const;
    DualInterval operator-(const double rhs) const;
    friend DualInterval operator-(const double lhs, const DualInterval& rhs);
    DualInterval operator-() const;

    DualInterval operator*(const DualInterval& rhs) const;
    DualInterval operator*(const double rhs) const;
    friend DualInterval operator*(const double lhs, const DualInterval& rhs);

    DualInterval operator/(const DualInterval& rhs) const;
    DualInterval operator/(const double rhs) const;

    DualInterval& operator+=(const DualInterval& rhs);
    DualInterval& operator+=(const double rhs);

    // join (union) of intervals
    DualInterval operator|(const DualInterval& rhs) const;

    friend DualInterval sin(const DualInterval& x);
    friend DualInterval cos(const DualInterval& x);
    friend DualInterval exp(const DualInterval& x);
    friend DualInterval log(const DualInterval& x);
    friend DualInterval sqrt(const DualInterval& x);
    friend DualInterval tanh(const DualInterval& x);
    friend DualInterval atan(const DualInterval& x);
    friend DualInterval relu(const DualInterval& x);
    friend DualInterval abs(const DualInterval& x);
    friend DualInterval logistic(const DualInterval& x);
    friend DualInterval max(const DualInterval& x, const DualInterval& y);
    friend DualInterval min(const DualInterval& x, const DualInterval& y);

    friend std::ostream& operator<<(std::ostream& os, const DualInterval& di);

    bool isEmpty() const;
    void setReal(Interval real_);
    void setReal(double a, double b);
    void setDual(Interval dual_);
    void setDual(double a, double b);
    Interval getReal() const;
    Interval getDual() const;
};

typedef DualInterval DI;

void DualInterval::setReal(Interval real_) {
    this->real = real_;
}

void DualInterval::setReal(double a, double b) {
    this->real = Interval(a, b);
}

void DualInterval::setDual(Interval dual_) {
    this->dual = dual_;
}

void DualInterval::setDual(double a, double b) {
    this->dual = Interval(a, b);
}

Interval DualInterval::getReal() const {
    return this->real;
}
Interval DualInterval::getDual() const {
    return this->dual;
}

bool DualInterval::isEmpty() const {
    return (empty(real) && empty(dual));
}

std::ostream& operator<<(std::ostream& os, const DualInterval& di) {
    os << di.real << " + " << di.dual << "\u03B5";
    return os;
}

// join operator
DI DualInterval::operator|(const DI& rhs) const {
    if (rhs.isEmpty()) {
        return DI(this->real.lower(), this->real.upper(), this->dual.lower(), this->dual.upper());
    }

    if (this->isEmpty()) {
        return DI(rhs.real.lower(), rhs.real.upper(), rhs.dual.lower(), rhs.dual.upper());
    }

    return DI(hull(this->real, rhs.real), hull(this->dual, rhs.dual));
}

// internal function only used within this header
Interval add_intervals(const Interval& a, const Interval& b) {
    Interval i = a + b;

    #ifdef SOUND
        double maxA = fmax(fabs(a.lower()), fabs(a.upper()));
        double maxB = fmax(fabs(b.lower()), fabs(b.upper()));
        Interval tmp = Interval(-maxA * ulp, maxA * ulp) + Interval(-maxB * ulp, maxB * ulp) + Interval(-min_denormal, min_denormal);
        return i + tmp;
    #endif

    return i;
}

// internal function only used within this header
Interval mul_intervals(const Interval& a, const Interval& b) {
    Interval i = a * b;

    #ifdef SOUND
        double maxB = fmax(fabs(b.lower()), fabs(b.upper()));
        Interval tmp = a * Interval(-maxB * ulp, maxB * ulp) + Interval(-min_denormal, min_denormal);
        return i + tmp;
    #endif

    return i;
}

// internal function only used within this header
Interval square_interval(const Interval& a) {
    Interval i = square(a);

    #ifdef SOUND
        double maxB = fmax(fabs(a.lower()), fabs(a.upper()));
        Interval tmp = a * Interval(-maxB * ulp, maxB * ulp) + Interval(-min_denormal, min_denormal);
        return i + tmp;
    #endif

    return i;
}

// internal function only used within this header
Interval div_intervals(const Interval& a, const Interval& b) {
    Interval i = a / b;

    #ifdef SOUND
        double maxA = fmax(fabs(a.lower()), fabs(a.upper()));
        Interval tmp = Interval(-maxA * ulp, maxA * ulp) / b + Interval(-min_denormal, min_denormal);
        return i + tmp;
    #endif

    return i;
}

// addition
DI DualInterval::operator+(const DI& rhs) const {
    assert(!this->isEmpty() && !rhs.isEmpty());

    DI temp;
    temp.real = add_intervals(this->real, rhs.real);
    temp.dual = add_intervals(this->dual, rhs.dual);
    return temp;
}

// addition with a scalar (this automatically casts the scalar to a DualInterval)
DI DualInterval::operator+(const double rhs) const {
    return *this + DI(rhs);
}

DI operator+(const double lhs, const DI& rhs) {
    return DI(lhs) + rhs;
}

// subtraction
DI DualInterval::operator-(const DI& rhs) const {
    return *this + (-rhs);
}

DI DualInterval::operator-(const double rhs) const {
    return *this - DI(rhs);
}

DI operator-(const double lhs, const DI& rhs) {
    return DI(lhs) - rhs;
}

// negation
DI DualInterval::operator-() const {
    return DI(-real.upper(), -real.lower(), -dual.upper(), -dual.lower());
}

// multiplication
DI DualInterval::operator*(const DI& rhs) const {
    assert(!this->isEmpty() && !rhs.isEmpty());

    DI temp;
    temp.real = mul_intervals(this->real, rhs.real);
    temp.dual = add_intervals(mul_intervals(this->real, rhs.dual), mul_intervals(this->dual, rhs.real));
    return temp;
}

DI DualInterval::operator*(const double rhs) const {
    return *this * DI(rhs);
}

DI operator*(const double lhs, const DI& rhs) {
    return DI(lhs) * rhs;
}

// division
DI DualInterval::operator/(const DI& rhs) const {
    assert(!this->isEmpty() && !rhs.isEmpty());

    DI temp;
    temp.real = div_intervals(this->real, rhs.real);
    temp.dual = div_intervals(add_intervals(mul_intervals(this->dual, rhs.real), -mul_intervals(this->real, rhs.dual)), square_interval(rhs.real));
    return temp;
}

DI DualInterval::operator/(const double rhs) const {
    return *this / DI(rhs);
}

// increment
DI& DualInterval::operator+=(const DI& rhs) {
    assert(!this->isEmpty() && !rhs.isEmpty());

    this->real = add_intervals(this->real, rhs.real);
    this->dual = add_intervals(this->dual, rhs.dual);
    return *this;
}

DI& DualInterval::operator+=(double rhs) {
    return *this += DI(rhs);
}

////////////////////////////////////////////////////////////////////////////////
//
// FUNCTIONS OF SCALAR DUAL INTERVALS
//
////////////////////////////////////////////////////////////////////////////////

DI exp(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = exp(x.real);
    temp.dual = temp.real * x.dual;
    return temp;
}

DI log(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = log(x.real);
    temp.dual = x.dual / x.real;
    return temp;
}

DI sqrt(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = sqrt(x.real);
    temp.dual = (x.dual / temp.real) * (0.5);
    return temp;
}

DI sin(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = sin(x.real);
    temp.dual = x.dual * cos(x.real);
    return temp;
}

DI cos(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = cos(x.real);
    temp.dual = -x.dual * sin(x.real);
    return temp;
}

DI relu(const DI& x) {
    assert(!x.isEmpty());

    if (x.real.lower() > 0) {
        return DI(x.real.lower(), x.real.upper(), x.dual.lower(), x.dual.upper());
    } else if (x.real.upper() < 0) {
        return DI{};  // empty constructors returns DI(0, 0, 0, 0)
    }

    // note how we restrict the real part by intersecting it with the set {x : x >0}
    DI true_branch(0, x.real.upper(), x.dual.lower(), x.dual.upper());
    DI false_branch{};
    return (true_branch | false_branch);  // use the join
}

DI max(const DI& x, const DI& y) {
    assert(!x.isEmpty() && !y.isEmpty());

    DI temp;
    if (x.real.lower() > y.real.upper()) {
        temp.real = x.real;
        temp.dual = x.dual;
    } else if (x.real.upper() < y.real.lower()) {
        temp.real = y.real;
        temp.dual = y.dual;
    } else {
        temp.real = max(x.real, y.real);
        temp.dual = hull(x.dual, y.dual);
    }
    return temp;
}

DI min(const DI& x, const DI& y) {
    return -max(-x, -y);
}

DI tanh(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = tanh(x.real);
    temp.dual = (1. - square(tanh(x.real))) * x.dual;
    return temp;
}

DI atan(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = atan(x.real);
    temp.dual = (1. / (1. + square(x.real))) * x.dual;
    return temp;
}

DI logistic(const DI& x) {
    assert(!x.isEmpty());

    DI temp;
    temp.real = (0.5 * tanh(x.real / 2.)) + 0.5;
    temp.dual = (0.25 * (1. - square(tanh(x.real / 2.)))) * x.dual;
    return temp;
}

DI abs(const DI& x) {
    assert(!x.isEmpty());

    if (x.real.upper() < 0.) {
        DI temp;
        temp.real = -x.real;
        temp.dual = -x.dual;
        return temp;
    } else if (x.real.lower() > 0.) {
        DI temp;
        temp.real = x.real;
        temp.dual = x.dual;
        return temp;
    } else {
        //note how we restrict the real part by intersecting it with the set {x : x > 0}
        DI positive_branch(0, x.real.upper(), x.dual.lower(), x.dual.upper());

        //note how we restrict the real part by intersecting it with the set {x : x < 0}
        Interval neg_branch_dual = -x.dual;
        DI negative_branch(0, -x.real.lower(), neg_branch_dual.lower(), neg_branch_dual.upper());
        return (positive_branch | negative_branch);
    }
}
