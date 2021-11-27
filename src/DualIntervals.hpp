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

    DualInterval(double rl, double ru, double dl, double du) {
        real = Interval(rl, ru);
        dual = Interval(dl, du);
    }

    DualInterval(const Interval& ri, const Interval& di) {
        real = ri;
        dual = di;
    }

    DualInterval(double r) {
        real = Interval(r, r);
        dual = Interval(0, 0);
    }

    DualInterval() {
        real = Interval(0, 0);
        dual = Interval(0, 0);
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

    DualInterval operator|(const DualInterval& rhs) const;

    friend DualInterval exp(const DualInterval& x);
    friend DualInterval log(const DualInterval& x);
    friend DualInterval sqrt(const DualInterval& x);
    friend DualInterval tanh(const DualInterval& x);
    friend DualInterval atan(const DualInterval& x);
    friend DualInterval logistic(const DualInterval& x);
    friend DualInterval sin(const DualInterval& x);
    friend DualInterval cos(const DualInterval& x);
    friend DualInterval abs(const DualInterval& x);
    friend DualInterval relu(const DualInterval& x);
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

bool DualInterval::isEmpty() const {
    return (empty(real) && empty(dual));
}

void DualInterval::setReal(Interval real_) {
    real = real_;
}

void DualInterval::setReal(double a, double b) {
    real = Interval(a, b);
}

void DualInterval::setDual(Interval dual_) {
    dual = dual_;
}

void DualInterval::setDual(double a, double b) {
    dual = Interval(a, b);
}

Interval DualInterval::getReal() const {
    return real;
}
Interval DualInterval::getDual() const {
    return dual;
}

std::ostream& operator<<(std::ostream& os, const DualInterval& di) {
    os << di.real << " + " << di.dual << "\u03B5";
    return os;
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
    assert(!isEmpty() && !rhs.isEmpty());

    Interval r = add_intervals(real, rhs.real);
    Interval d = add_intervals(dual, rhs.dual);
    return DI(r, d);
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
    assert(!isEmpty() && !rhs.isEmpty());

    Interval r = mul_intervals(real, rhs.real);
    Interval d = add_intervals(mul_intervals(real, rhs.dual), mul_intervals(dual, rhs.real));
    return DI(r, d);
}

DI DualInterval::operator*(const double rhs) const {
    return *this * DI(rhs);
}

DI operator*(const double lhs, const DI& rhs) {
    return DI(lhs) * rhs;
}

// division
DI DualInterval::operator/(const DI& rhs) const {
    assert(!isEmpty() && !rhs.isEmpty());

    Interval r = div_intervals(real, rhs.real);
    Interval d = div_intervals(add_intervals(mul_intervals(dual, rhs.real), -mul_intervals(real, rhs.dual)), square_interval(rhs.real));
    return DI(r, d);
}

DI DualInterval::operator/(const double rhs) const {
    return *this / DI(rhs);
}

// increment
DI& DualInterval::operator+=(const DI& rhs) {
    assert(!isEmpty() && !rhs.isEmpty());

    real = add_intervals(real, rhs.real);
    dual = add_intervals(dual, rhs.dual);
    return *this;
}

DI& DualInterval::operator+=(double rhs) {
    return *this += DI(rhs);
}

// join (union)
DI DualInterval::operator|(const DI& rhs) const {
    if (rhs.isEmpty()) {
        return DI(real.lower(), real.upper(), dual.lower(), dual.upper());
    }

    if (isEmpty()) {
        return DI(rhs.real.lower(), rhs.real.upper(), rhs.dual.lower(), rhs.dual.upper());
    }

    return DI(hull(real, rhs.real), hull(dual, rhs.dual));
}

////////////////////////////////////////////////////////////////////////////////
//
// FUNCTIONS OF SCALAR DUAL INTERVALS
//
////////////////////////////////////////////////////////////////////////////////

DI exp(const DI& x) {
    assert(!x.isEmpty());

    Interval r = exp(x.real);
    Interval d = r * x.dual;
    return DI(r, d);
}

DI log(const DI& x) {
    assert(!x.isEmpty());
    return DI(log(x.real), x.dual / x.real);
}

DI sqrt(const DI& x) {
    assert(!x.isEmpty());

    Interval r = sqrt(x.real);
    Interval d = x.dual / r * 0.5;
    return DI(r, d);
}

DI tanh(const DI& x) {
    assert(!x.isEmpty());

    Interval r = tanh(x.real);
    Interval d = (1. - square(r)) * x.dual;
    return DI(r, d);
}

DI atan(const DI& x) {
    assert(!x.isEmpty());
    return DI(atan(x.real), 1. / (1. + square(x.real)) * x.dual);
}

DI logistic(const DI& x) {
    assert(!x.isEmpty());

    Interval t = tanh(x.real / 2.);
    Interval r = 0.5 * t + 0.5;
    Interval d = 0.25 * (1. - square(t)) * x.dual;
    return DI(r, d);
}

DI sin(const DI& x) {
    assert(!x.isEmpty());
    return DI(sin(x.real), cos(x.real) * x.dual);
}

DI cos(const DI& x) {
    assert(!x.isEmpty());
    return DI(cos(x.real), -sin(x.real) * x.dual);
}

DI abs(const DI& x) {
    assert(!x.isEmpty());

    if (x.real.upper() < 0) {
        return -x;
    } else if (x.real.lower() > 0) {
        return x;
    } else {
        DI positive_branch(0, x.real.upper(), x.dual.lower(), x.dual.upper());
        DI negative_branch(0, -x.real.lower(), -x.dual.upper(), -x.dual.lower());
        return (positive_branch | negative_branch);
    }
}

DI max(const DI& x, const DI& y) {
    assert(!x.isEmpty() && !y.isEmpty());

    if (x.real.lower() > y.real.upper()) {
        return x;
    } else if (x.real.upper() < y.real.lower()) {
        return y;
    }
    return DI(max(x.real, y.real), hull(x.dual, y.dual));
}

DI min(const DI& x, const DI& y) {
    return -max(-x, -y);
}

DI relu(const DI& x) {
    return max(x, 0);
}
