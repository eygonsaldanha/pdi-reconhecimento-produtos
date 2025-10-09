"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CarIcon as CartIcon, Trash2, Plus, Minus } from "lucide-react"
import { useCart } from "@/components/cart-context"

export function ShoppingCart() {
  const { items, removeFromCart, updateQuantity, getTotalPrice, getTotalItems } = useCart()

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL",
    }).format(price)
  }

  return (
    <div className="w-96 bg-sidebar border-l border-sidebar-border p-6 overflow-y-auto">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-sidebar-foreground flex items-center">
            <CartIcon className="w-6 h-6 mr-2" />
            Carrinho
          </h2>
          {getTotalItems() > 0 && (
            <Badge className="bg-sidebar-primary text-sidebar-primary-foreground">{getTotalItems()}</Badge>
          )}
        </div>

        {/* Cart Items */}
        <div className="space-y-4">
          {items.length === 0 ? (
            <Card className="p-6 text-center">
              <div className="text-muted-foreground space-y-2">
                <CartIcon className="w-12 h-12 mx-auto opacity-50" />
                <p>Seu carrinho está vazio</p>
                <p className="text-sm">Adicione produtos para começar</p>
              </div>
            </Card>
          ) : (
            items.map((item) => (
              <Card key={item.id} className="p-4">
                <div className="space-y-3">
                  {/* Product Image and Info */}
                  <div className="flex space-x-3">
                    <div className="w-16 h-16 rounded-md overflow-hidden flex-shrink-0">
                      <img
                        src={item.image || "/placeholder.svg"}
                        alt={item.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-card-foreground truncate">{item.name}</h3>
                      <p className="text-primary font-bold">{formatPrice(item.price)}</p>
                    </div>
                  </div>

                  {/* Quantity Controls */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => updateQuantity(item.id, Math.max(0, item.quantity - 1))}
                        className="w-8 h-8 p-0"
                      >
                        <Minus className="w-3 h-3" />
                      </Button>
                      <span className="w-8 text-center font-medium">{item.quantity}</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => updateQuantity(item.id, item.quantity + 1)}
                        className="w-8 h-8 p-0"
                      >
                        <Plus className="w-3 h-3" />
                      </Button>
                    </div>

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFromCart(item.id)}
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>

                  {/* Item Total */}
                  <div className="text-right">
                    <p className="text-sm text-muted-foreground">
                      Subtotal:{" "}
                      <span className="font-medium text-foreground">{formatPrice(item.price * item.quantity)}</span>
                    </p>
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>

        {/* Cart Summary */}
        {items.length > 0 && (
          <Card className="p-4 bg-sidebar-accent">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-lg font-medium text-sidebar-accent-foreground">Total:</span>
                <span className="text-2xl font-bold text-sidebar-accent-foreground">
                  {formatPrice(getTotalPrice())}
                </span>
              </div>

              <Button
                className="w-full bg-sidebar-primary hover:bg-sidebar-primary/90 text-sidebar-primary-foreground"
                size="lg"
              >
                Finalizar Compra
              </Button>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
