# Lesson 1: Swift Programming Language for Healthcare Apps

## 🎯 Purpose
Swift is Apple's modern programming language designed for safety, performance, and expressiveness. For a TPA healthcare app, Swift's type safety and error handling features are crucial for maintaining data integrity and HIPAA compliance.

## 💼 Business Logic
In healthcare apps, data accuracy is paramount. Swift's features help:
- Prevent runtime crashes that could expose PHI
- Ensure data types match healthcare standards (FHIR)
- Handle network failures gracefully
- Maintain audit trails with proper error logging

## 🏥 Healthcare-Specific Swift Concepts

### 1. Type Safety for Medical Data

```swift
// NEVER use generic types for healthcare data
// BAD: var claimAmount: Any = 1250.50

// GOOD: Use specific types with validation
struct ClaimAmount {
    private let value: Decimal
    let currency: Currency
    
    init(value: Decimal, currency: Currency = .usd) throws {
        guard value >= 0 else {
            throw ValidationError.negativeAmount
        }
        self.value = value
        self.currency = currency
    }
    
    var formatted: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = currency.rawValue
        return formatter.string(from: value as NSNumber) ?? ""
    }
}

enum Currency: String {
    case usd = "USD"
    case mxn = "MXN" // For bilingual support
}
```

### 2. Optionals for Healthcare Data Integrity

```swift
// Healthcare data often has optional fields
struct PatientProfile {
    let memberId: String // Required
    let firstName: String // Required
    let lastName: String // Required
    let middleName: String? // Optional
    let preferredName: String? // Optional
    let dateOfBirth: Date // Required for eligibility
    let ssn: String? // Optional, encrypted when present
    
    // Computed property for display name
    var displayName: String {
        if let preferred = preferredName {
            return preferred
        }
        return "\(firstName) \(lastName)"
    }
    
    // Safe SSN display
    var maskedSSN: String? {
        guard let ssn = ssn, ssn.count >= 4 else { return nil }
        return "***-**-\(ssn.suffix(4))"
    }
}
```

### 3. Error Handling for API Failures

```swift
// Define comprehensive healthcare errors
enum HealthcareAPIError: LocalizedError {
    case networkUnavailable
    case authenticationExpired
    case insufficientPermissions
    case claimNotFound(claimId: String)
    case providerNotInNetwork(providerId: String)
    case hipaaAuditFailure
    
    var errorDescription: String? {
        switch self {
        case .networkUnavailable:
            return "Unable to connect. Please check your internet connection."
        case .authenticationExpired:
            return "Your session has expired. Please log in again."
        case .insufficientPermissions:
            return "You don't have permission to access this information."
        case .claimNotFound(let id):
            return "Claim \(id) not found."
        case .providerNotInNetwork(let id):
            return "Provider \(id) is not in your network."
        case .hipaaAuditFailure:
            return "Security audit failed. Please contact support."
        }
    }
}

// Robust API call with retry logic
class ClaimsService {
    func fetchClaim(id: String) async throws -> Claim {
        var retryCount = 0
        let maxRetries = 3
        
        while retryCount < maxRetries {
            do {
                // Log attempt for HIPAA audit
                await AuditLogger.log(event: .claimAccess(id: id, attempt: retryCount + 1))
                
                let claim = try await performAPICall(claimId: id)
                
                // Log success
                await AuditLogger.log(event: .claimAccessSuccess(id: id))
                
                return claim
            } catch {
                retryCount += 1
                
                if retryCount == maxRetries {
                    // Log failure
                    await AuditLogger.log(event: .claimAccessFailure(id: id, error: error))
                    throw error
                }
                
                // Exponential backoff
                try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(retryCount)) * 1_000_000_000))
            }
        }
        
        throw HealthcareAPIError.networkUnavailable
    }
}
```

### 4. Async/Await for Healthcare Operations

```swift
// Modern async patterns for healthcare workflows
actor HealthDataSynchronizer {
    private var isSyncing = false
    
    func syncAllData(for memberId: String) async throws {
        guard !isSyncing else { return }
        isSyncing = true
        
        defer { isSyncing = false }
        
        // Parallel fetch for better performance
        async let claims = fetchClaims(for: memberId)
        async let benefits = fetchBenefits(for: memberId)
        async let providers = fetchProviders(for: memberId)
        async let idCard = fetchIDCard(for: memberId)
        
        // Wait for all to complete
        let (claimsData, benefitsData, providersData, idCardData) = try await (claims, benefits, providers, idCard)
        
        // Store in encrypted Core Data
        try await SecureDataStore.shared.store(
            claims: claimsData,
            benefits: benefitsData,
            providers: providersData,
            idCard: idCardData,
            for: memberId
        )
    }
}
```

### 5. Property Wrappers for Security

```swift
// Custom property wrapper for PHI encryption
@propertyWrapper
struct EncryptedPHI {
    private var value: String?
    
    var wrappedValue: String? {
        get {
            guard let encrypted = value else { return nil }
            return CryptoManager.shared.decrypt(encrypted)
        }
        set {
            value = newValue.map { CryptoManager.shared.encrypt($0) }
        }
    }
}

// Usage in models
struct MedicalRecord {
    let recordId: String
    @EncryptedPHI var diagnosis: String?
    @EncryptedPHI var medications: String?
    @EncryptedPHI var allergies: String?
    let lastUpdated: Date
}
```

### 6. Enums for Healthcare Standards

```swift
// FHIR-compliant status enums
enum ClaimStatus: String, CaseIterable {
    case draft = "draft"
    case active = "active"
    case suspended = "suspended"
    case completed = "completed"
    case enteredInError = "entered-in-error"
    case cancelled = "cancelled"
    
    var displayText: String {
        switch self {
        case .draft: return "Draft"
        case .active: return "In Progress"
        case .suspended: return "On Hold"
        case .completed: return "Completed"
        case .enteredInError: return "Error"
        case .cancelled: return "Cancelled"
        }
    }
    
    var color: Color {
        switch self {
        case .draft: return .gray
        case .active: return .blue
        case .suspended: return .orange
        case .completed: return .green
        case .enteredInError: return .red
        case .cancelled: return .gray
        }
    }
}
```

### 7. Protocols for Extensibility

```swift
// Protocol for all healthcare entities
protocol HealthcareEntity {
    var id: String { get }
    var lastModified: Date { get }
    var isActive: Bool { get }
    
    func validate() throws
    func auditLog() -> AuditEntry
}

// Protocol for searchable items
protocol Searchable {
    func matches(query: String) -> Bool
    var searchableText: String { get }
}

// Combine protocols for providers
struct Provider: HealthcareEntity, Searchable {
    let id: String
    let name: String
    let specialty: String
    let networkStatus: NetworkStatus
    let lastModified: Date
    var isActive: Bool
    
    func validate() throws {
        guard !name.isEmpty else {
            throw ValidationError.emptyField("name")
        }
        guard !specialty.isEmpty else {
            throw ValidationError.emptyField("specialty")
        }
    }
    
    func auditLog() -> AuditEntry {
        AuditEntry(
            entityType: "Provider",
            entityId: id,
            action: .access,
            timestamp: Date()
        )
    }
    
    func matches(query: String) -> Bool {
        searchableText.localizedCaseInsensitiveContains(query)
    }
    
    var searchableText: String {
        "\(name) \(specialty) \(networkStatus.rawValue)"
    }
}
```

## 🛡️ Security Best Practices in Swift

### 1. Never Store Sensitive Data in UserDefaults
```swift
// BAD
UserDefaults.standard.set(ssn, forKey: "userSSN")

// GOOD
try KeychainManager.shared.store(ssn, for: .ssn)
```

### 2. Use Codable with Custom Encoding
```swift
struct SecureClaim: Codable {
    let id: String
    let amount: Decimal
    private let encryptedDetails: String
    
    enum CodingKeys: String, CodingKey {
        case id = "claim_id"
        case amount = "claim_amount"
        case encryptedDetails = "encrypted_data"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        amount = try container.decode(Decimal.self, forKey: .amount)
        encryptedDetails = try container.decode(String.self, forKey: .encryptedDetails)
        
        // Validate after decoding
        try validate()
    }
    
    private func validate() throws {
        guard amount >= 0 else {
            throw ValidationError.invalidClaimAmount
        }
    }
}
```

## 📱 Memory Management for Healthcare Apps

```swift
// Proper image handling for claim photos
class ClaimPhotoManager {
    private let cache = NSCache<NSString, UIImage>()
    
    init() {
        // Configure cache for medical images
        cache.countLimit = 20 // Limit number of images
        cache.totalCostLimit = 100 * 1024 * 1024 // 100MB limit
    }
    
    func loadPhoto(for claimId: String) async throws -> UIImage {
        // Check cache first
        if let cached = cache.object(forKey: claimId as NSString) {
            return cached
        }
        
        // Load from secure storage
        let imageData = try await SecureFileManager.shared.loadImage(claimId: claimId)
        
        guard let image = UIImage(data: imageData) else {
            throw ImageError.invalidFormat
        }
        
        // Resize if needed to prevent memory issues
        let resized = image.resized(toMaxDimension: 1024)
        
        // Cache the resized image
        cache.setObject(resized, forKey: claimId as NSString)
        
        return resized
    }
}
```

## 🌐 Bilingual Support

```swift
// Localization for English/Spanish
extension String {
    var localized: String {
        NSLocalizedString(self, comment: "")
    }
    
    func localized(with arguments: CVarArg...) -> String {
        String(format: self.localized, arguments: arguments)
    }
}

// Usage
struct LocalizedMessages {
    static let welcomeMessage = "welcome_message".localized
    static let claimSubmitted = "claim_submitted".localized
    
    static func claimAmount(_ amount: String) -> String {
        "claim_amount_format".localized(with: amount)
    }
}
```

## 🎯 Exercise: Build a Secure Member Model

Create a `Member` struct that:
1. Has required fields for ID, name, DOB
2. Has optional fields for SSN, email
3. Implements proper validation
4. Provides secure display methods
5. Conforms to Codable
6. Includes audit logging

## 🔑 Key Takeaways

1. **Type Safety**: Use Swift's type system to prevent healthcare data errors
2. **Error Handling**: Implement comprehensive error handling for all API calls
3. **Security First**: Encrypt sensitive data, use Keychain, implement audit logging
4. **Memory Aware**: Handle medical images and large datasets efficiently
5. **Async Operations**: Use modern concurrency for responsive UI
6. **Localization**: Build for bilingual support from the start

## 📚 Next Steps

In Lesson 2, we'll build upon these Swift fundamentals to create beautiful, accessible healthcare UI components using SwiftUI, including:
- Custom ID card views with security features
- Accessible forms for claim submission
- Charts for healthcare analytics
- Biometric authentication flows

Ready to move on to Lesson 2: SwiftUI Framework for Healthcare UI?