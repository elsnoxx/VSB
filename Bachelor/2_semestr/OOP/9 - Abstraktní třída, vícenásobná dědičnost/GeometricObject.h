#pragma once
#ifndef GEOMETRIC_OBJECT_ABSTRACT_H
#define GEOMETRIC_OBJECT_ABSTRACT_H

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class GeometricObjectAbstract
 * @brief Abstraktní třída reprezentující geometrický objekt.
 */
class GeometricObjectAbstract {
public:
    /**
     * @brief Čistě virtuální metoda pro výpočet obvodu geometrického objektu.
     * @return Obvod geometrického objektu.
     */
    virtual double obvod() const = 0;

    /**
     * @brief Čistě virtuální metoda pro výpočet obsahu geometrického objektu.
     * @return Obsah geometrického objektu.
     */
    virtual double obsah() const = 0;

    /**
     * @brief Virtuální destruktor pro třídu GeometricObjectAbstract.
     */
    virtual ~GeometricObjectAbstract() = default;
};

/**
 * @class Circle
 * @brief Třída představující kruh jako geometrický objekt.
 */
class Circle : public GeometricObjectAbstract {
private:
    double radius;  /**< Poloměr kruhu */

public:
    /**
     * @brief Konstruktor pro inicializaci kruhu s daným poloměrem.
     * @param r Poloměr kruhu.
     */
    Circle(double r);

    double obvod() const override;
    double obsah() const override;
};

/**
 * @class Rectangle
 * @brief Třída představující obdélník jako geometrický objekt.
 */
class Rectangle : public GeometricObjectAbstract {
private:
    double width, height;  /**< Šířka a výška obdélníku */

public:
    /**
     * @brief Konstruktor pro inicializaci obdélníku s danou šířkou a výškou.
     * @param w Šířka obdélníku.
     * @param h Výška obdélníku.
     */
    Rectangle(double w, double h);

    double obvod() const override;
    double obsah() const override;
};

/**
 * @class Square
 * @brief Třída představující čtverec jako geometrický objekt.
 */
class Square : public GeometricObjectAbstract {
private:
    double side;  /**< Délka strany čtverce */

public:
    /**
     * @brief Konstruktor pro inicializaci čtverce s danou délkou strany.
     * @param s Délka strany čtverce.
     */
    Square(double s);

    double obvod() const override;
    double obsah() const override;
};

/**
 * @class Triangle
 * @brief Třída představující trojúhelník jako geometrický objekt.
 */
class Triangle : public GeometricObjectAbstract {
private:
    double side1, side2, side3;  /**< Délky stran trojúhelníka */

public:
    /**
     * @brief Konstruktor pro inicializaci trojúhelníka s danými délkami stran.
     * @param s1 Délka první strany trojúhelníka.
     * @param s2 Délka druhé strany trojúhelníka.
     * @param s3 Délka třetí strany trojúhelníka.
     */
    Triangle(double s1, double s2, double s3);

    double obvod() const override;
    double obsah() const override;
};

#endif // GEOMETRIC_OBJECT_ABSTRACT_H
