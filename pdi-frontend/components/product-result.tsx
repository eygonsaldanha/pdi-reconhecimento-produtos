"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ShoppingCart, X, RotateCcw } from "lucide-react"
import { useCart } from "@/components/cart-context"

interface ProductResultProps {
  product: {
    id: string
    name: string
    price: number
    image: string
    confidence: number
  }
  onReset: () => void
}

export function ProductResult({ product, onReset }: ProductResultProps) {
  const { addToCart } = useCart()

  const handleAddToCart = () => {
    addToCart({
      id: product.id,
      name: product.name,
      price: product.price,
      image: product.image,
      quantity: 1,
    })
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL",
    }).format(price)
  }

  return (
    <Card className="w-full max-w-2xl p-6">
      <div className="space-y-6">
        {/* Product Image */}
        <div className="relative mx-auto w-64 h-64 rounded-lg overflow-hidden">
          <img src={product.image || "/placeholder.svg"} alt={product.name} className="w-full h-full object-cover" />
          <Badge className="absolute top-2 right-2 bg-accent text-accent-foreground">
            {Math.round(product.confidence * 100)}% confiança
          </Badge>
        </div>

        {/* Product Info */}
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold text-card-foreground">{product.name}</h2>
          <p className="text-4xl font-bold text-primary">{formatPrice(product.price)}</p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4">
          <Button
            variant="outline"
            onClick={onReset}
            className="flex-1 border-destructive text-destructive hover:bg-destructive hover:text-destructive-foreground bg-transparent"
          >
            <X className="w-4 h-4 mr-2" />
            Produto Incorreto
          </Button>

          <Button onClick={handleAddToCart} className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground">
            <ShoppingCart className="w-4 h-4 mr-2" />
            Adicionar ao Carrinho
          </Button>
        </div>

        {/* Reset Button */}
        <div className="text-center pt-4 border-t">
          <Button variant="ghost" onClick={onReset} className="text-muted-foreground hover:text-foreground">
            <RotateCcw className="w-4 h-4 mr-2" />
            Analisar Outro Produto
          </Button>
        </div>
      </div>
    </Card>
  )
}
