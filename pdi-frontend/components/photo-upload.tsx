"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Camera, Upload, Loader2, Sparkles } from "lucide-react"

interface PhotoUploadProps {
  onPhotoAnalyzed: (product: {
    id: string
    name: string
    price: number
    image: string
    confidence: number
  }) => void
}

export function PhotoUpload({ onPhotoAnalyzed }: PhotoUploadProps) {
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
      { name: "Banana Prata", price: 5.99, confidence: 0.95 },
      { name: "Leite Integral 1L", price: 4.89, confidence: 0.92 },
      { name: "Pão de Açúcar Francês", price: 8.5, confidence: 0.88 },
      { name: "Detergente Ypê", price: 2.49, confidence: 0.91 },
      { name: "Arroz Branco 5kg", price: 24.99, confidence: 0.89 },
      { name: "Sabonete Dove", price: 3.79, confidence: 0.93 },
      { name: "Coca-Cola 2L", price: 7.99, confidence: 0.96 },
      { name: "Ovos Brancos 12un", price: 8.99, confidence: 0.9 },
      { name: "Papel Higiênico Neve", price: 12.5, confidence: 0.87 },
      { name: "Feijão Preto 1kg", price: 6.89, confidence: 0.94 },
      { name: "Maçã Gala", price: 7.99, confidence: 0.92 },
      { name: "Açúcar Cristal 1kg", price: 3.99, confidence: 0.88 },
      { name: "Amaciante Comfort", price: 5.49, confidence: 0.85 },
      { name: "Margarina Qualy", price: 4.29, confidence: 0.91 },
      { name: "Macarrão Espaguete", price: 3.5, confidence: 0.89 },
    ];

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
    <Card className="w-full max-w-2xl p-8 glass border-2 border-primary/10 shadow-2xl hover:shadow-primary/10 transition-all duration-500 hover:scale-[1.02]">
      <div className="text-center space-y-8">
        {!uploadedImage && !isAnalyzing && (
          <>
            <div className="mx-auto w-32 h-32 gradient-primary rounded-full flex items-center justify-center shadow-lg animate-float">
              <Camera className="w-16 h-16 text-white drop-shadow-lg" />
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
              className="gradient-primary hover:scale-105 transition-all duration-300 text-white px-10 py-4 text-lg font-semibold shadow-lg hover:shadow-primary/25 group"
            >
              <Upload className="w-6 h-6 mr-3 group-hover:rotate-12 transition-transform duration-300" />
              Selecionar Foto
              <Sparkles className="w-5 h-5 ml-2 opacity-70" />
            </Button>
          </>
        )}

        {uploadedImage && isAnalyzing && (
          <>
            <div className="mx-auto w-64 h-64 rounded-2xl overflow-hidden shadow-2xl ring-4 ring-primary/20 animate-pulse">
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
