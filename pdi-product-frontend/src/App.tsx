"use client"

import type React from "react"

import { useState, useRef, createContext, useContext, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Camera, Upload, Loader2, Sparkles, ShoppingCart, X, RotateCcw, Trash2, Plus, Minus } from "lucide-react"

interface CartItem {
  id: string
  name: string
  price: number
  image: string
  quantity: number
}

interface CartContextType {
  items: CartItem[]
  addToCart: (item: Omit<CartItem, "quantity"> & { quantity: number }) => void
  removeFromCart: (id: string) => void
  updateQuantity: (id: string, quantity: number) => void
  getTotalPrice: () => number
  getTotalItems: () => number
}

const CartContext = createContext<CartContextType | undefined>(undefined)

function CartProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<CartItem[]>([])

  const addToCart = (newItem: Omit<CartItem, "quantity"> & { quantity: number }) => {
    setItems((currentItems) => {
      const existingItem = currentItems.find((item) => item.id === newItem.id)

      if (existingItem) {
        return currentItems.map((item) =>
          item.id === newItem.id ? { ...item, quantity: item.quantity + newItem.quantity } : item,
        )
      }

      return [...currentItems, newItem]
    })
  }

  const removeFromCart = (id: string) => {
    setItems((currentItems) => currentItems.filter((item) => item.id !== id))
  }

  const updateQuantity = (id: string, quantity: number) => {
    if (quantity <= 0) {
      removeFromCart(id)
      return
    }

    setItems((currentItems) => currentItems.map((item) => (item.id === id ? { ...item, quantity } : item)))
  }

  const getTotalPrice = () => {
    return items.reduce((total, item) => total + item.price * item.quantity, 0)
  }

  const getTotalItems = () => {
    return items.reduce((total, item) => total + item.quantity, 0)
  }

  return (
    <CartContext.Provider
      value={{
        items,
        addToCart,
        removeFromCart,
        updateQuantity,
        getTotalPrice,
        getTotalItems,
      }}
    >
      {children}
    </CartContext.Provider>
  )
}

function useCart() {
  const context = useContext(CartContext)
  if (context === undefined) {
    throw new Error("useCart must be used within a CartProvider")
  }
  return context
}

function PhotoUpload({
  onPhotoAnalyzed,
}: {
  onPhotoAnalyzed: (product: {
    id: string
    name: string
    price: number
    image: string
    confidence: number
  }) => void
}) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const imageUrl = e.target?.result as string
        setUploadedImage(imageUrl)
        analyzePhoto(imageUrl)
      }
      reader.readAsDataURL(file)
    }
  }

  const analyzePhoto = async (imageUrl: string) => {
    setIsAnalyzing(true)

    const mockProducts = [
      { name: "iPhone 15 Pro", price: 1299.99, confidence: 0.95 },
      { name: "Samsung Galaxy S24", price: 999.99, confidence: 0.92 },
      { name: "MacBook Air M3", price: 1499.99, confidence: 0.88 },
      { name: "AirPods Pro", price: 249.99, confidence: 0.91 },
      { name: 'iPad Pro 12.9"', price: 1099.99, confidence: 0.89 },
      { name: "Apple Watch Series 9", price: 399.99, confidence: 0.93 },
    ]

    await new Promise((resolve) => setTimeout(resolve, 2000))

    const randomProduct = mockProducts[Math.floor(Math.random() * mockProducts.length)]

    onPhotoAnalyzed({
      id: Math.random().toString(36).substr(2, 9),
      name: randomProduct.name,
      price: randomProduct.price,
      image: imageUrl,
      confidence: randomProduct.confidence,
    })

    setIsAnalyzing(false)
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <Card className="w-full max-w-2xl p-8 bg-card/50 backdrop-blur-xl border-2 border-primary/20 shadow-2xl hover:shadow-primary/20 transition-all duration-500 hover:scale-[1.02]">
      <div className="text-center space-y-8">
        {!uploadedImage && !isAnalyzing && (
          <>
            <div className="mx-auto w-32 h-32 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center shadow-lg animate-float">
              <Camera className="w-16 h-16 text-primary-foreground drop-shadow-lg" />
            </div>
            <div className="space-y-4">
              <h2 className="text-3xl font-bold text-card-foreground">Tire uma Foto do Produto</h2>
              <p className="text-muted-foreground text-lg leading-relaxed max-w-md mx-auto">
                Faça upload de uma imagem para identificar o produto automaticamente com nossa IA
              </p>
            </div>
            <Button
              onClick={handleUploadClick}
              size="lg"
              className="bg-gradient-to-r from-primary to-accent hover:scale-105 transition-all duration-300 text-primary-foreground px-10 py-4 text-lg font-semibold shadow-lg hover:shadow-primary/25 group"
            >
              <Upload className="w-6 h-6 mr-3 group-hover:rotate-12 transition-transform duration-300" />
              Selecionar Foto
              <Sparkles className="w-5 h-5 ml-2 opacity-70" />
            </Button>
          </>
        )}

        {uploadedImage && isAnalyzing && (
          <>
            <div className="mx-auto w-64 h-64 rounded-2xl overflow-hidden shadow-2xl ring-4 ring-primary/30 animate-pulse">
              <img
                src={uploadedImage || "/placeholder.svg"}
                alt="Produto enviado"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="space-y-6">
              <div className="flex items-center justify-center space-x-3">
                <div className="relative">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  <div className="absolute inset-0 w-8 h-8 border-2 border-primary/20 rounded-full animate-ping" />
                </div>
                <span className="text-xl font-semibold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Analisando produto...
                </span>
              </div>
              <p className="text-muted-foreground text-lg">Nossa IA está identificando o produto na imagem</p>
              <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-accent rounded-full animate-pulse"
                  style={{ width: "70%" }}
                />
              </div>
            </div>
          </>
        )}

        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
      </div>
    </Card>
  )
}

function ProductResult({
  product,
  onReset,
}: {
  product: {
    id: string
    name: string
    price: number
    image: string
    confidence: number
  }
  onReset: () => void
}) {
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
    <Card className="w-full max-w-2xl p-8 bg-card/50 backdrop-blur-xl border-2 border-primary/20 shadow-2xl">
      <div className="space-y-6">
        <div className="relative mx-auto w-64 h-64 rounded-2xl overflow-hidden shadow-xl">
          <img src={product.image || "/placeholder.svg"} alt={product.name} className="w-full h-full object-cover" />
          <Badge className="absolute top-3 right-3 bg-accent/90 text-accent-foreground backdrop-blur-sm">
            {Math.round(product.confidence * 100)}% confiança
          </Badge>
        </div>

        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold text-card-foreground">{product.name}</h2>
          <p className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            {formatPrice(product.price)}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
          <Button
            variant="outline"
            onClick={onReset}
            className="flex-1 border-destructive/50 text-destructive hover:bg-destructive hover:text-destructive-foreground bg-transparent backdrop-blur-sm"
          >
            <X className="w-4 h-4 mr-2" />
            Produto Incorreto
          </Button>

          <Button
            onClick={handleAddToCart}
            className="flex-1 bg-gradient-to-r from-primary to-accent hover:scale-105 transition-all duration-300 text-primary-foreground shadow-lg"
          >
            <ShoppingCart className="w-4 h-4 mr-2" />
            Adicionar ao Carrinho
          </Button>
        </div>

        <div className="text-center pt-4 border-t border-border/50">
          <Button variant="ghost" onClick={onReset} className="text-muted-foreground hover:text-foreground">
            <RotateCcw className="w-4 h-4 mr-2" />
            Analisar Outro Produto
          </Button>
        </div>
      </div>
    </Card>
  )
}

function ShoppingCartSidebar() {
  const { items, removeFromCart, updateQuantity, getTotalPrice, getTotalItems } = useCart()

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL",
    }).format(price)
  }

  return (
    <div className="w-96 bg-sidebar/50 backdrop-blur-xl border-l border-sidebar-border/50 p-6 overflow-y-auto">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-sidebar-foreground flex items-center">
            <ShoppingCart className="w-6 h-6 mr-2" />
            Carrinho
          </h2>
          {getTotalItems() > 0 && (
            <Badge className="bg-gradient-to-r from-primary to-accent text-primary-foreground">{getTotalItems()}</Badge>
          )}
        </div>

        <div className="space-y-4">
          {items.length === 0 ? (
            <Card className="p-6 text-center bg-card/30 backdrop-blur-sm">
              <div className="text-muted-foreground space-y-2">
                <ShoppingCart className="w-12 h-12 mx-auto opacity-50" />
                <p>Seu carrinho está vazio</p>
                <p className="text-sm">Adicione produtos para começar</p>
              </div>
            </Card>
          ) : (
            items.map((item) => (
              <Card key={item.id} className="p-4 bg-card/30 backdrop-blur-sm border-border/50">
                <div className="space-y-3">
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

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => updateQuantity(item.id, Math.max(0, item.quantity - 1))}
                        className="w-8 h-8 p-0 bg-background/50"
                      >
                        <Minus className="w-3 h-3" />
                      </Button>
                      <span className="w-8 text-center font-medium">{item.quantity}</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => updateQuantity(item.id, item.quantity + 1)}
                        className="w-8 h-8 p-0 bg-background/50"
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

        {items.length > 0 && (
          <Card className="p-4 bg-gradient-to-br from-sidebar-accent/50 to-sidebar-accent/30 backdrop-blur-sm">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-lg font-medium text-sidebar-accent-foreground">Total:</span>
                <span className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  {formatPrice(getTotalPrice())}
                </span>
              </div>

              <Button
                className="w-full bg-gradient-to-r from-primary to-accent hover:scale-105 transition-all duration-300 text-primary-foreground shadow-lg"
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

export default function Home() {
  const [analyzedProduct, setAnalyzedProduct] = useState<{
    id: string
    name: string
    price: number
    image: string
    confidence: number
  } | null>(null)

  const handlePhotoAnalyzed = (product: {
    id: string
    name: string
    price: number
    image: string
    confidence: number
  }) => {
    setAnalyzedProduct(product)
  }

  const handleResetProduct = () => {
    setAnalyzedProduct(null)
  }

  return (
    <CartProvider>
      <div className="dark min-h-screen bg-gradient-to-br from-background via-background to-muted/20 flex relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary/10 rounded-full blur-3xl animate-float" />
          <div
            className="absolute -bottom-40 -left-40 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-float"
            style={{ animationDelay: "3s" }}
          />
          <div
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-primary/5 rounded-full blur-3xl animate-float"
            style={{ animationDelay: "1.5s" }}
          />
        </div>

        {/* Conteúdo Principal */}
        <div className="flex-1 p-6 relative z-10">
          <div className="max-w-4xl mx-auto">
            <header className="text-center mb-12">
              <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent mb-4 animate-glow">
                Reconhecimento de Produtos
              </h1>
              <p className="text-muted-foreground text-xl max-w-2xl mx-auto leading-relaxed">
                Tire uma foto do produto e descubra o nome e preço instantaneamente com nossa IA avançada
              </p>
            </header>

            <div className="flex flex-col items-center space-y-8">
              {!analyzedProduct ? (
                <PhotoUpload onPhotoAnalyzed={handlePhotoAnalyzed} />
              ) : (
                <ProductResult product={analyzedProduct} onReset={handleResetProduct} />
              )}
            </div>
          </div>
        </div>

        {/* Sidebar do Carrinho */}
        <ShoppingCartSidebar />
      </div>
    </CartProvider>
  )
}
