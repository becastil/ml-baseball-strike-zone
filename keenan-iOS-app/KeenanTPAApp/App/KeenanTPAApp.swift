import SwiftUI

@main
struct KeenanTPAApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authManager)
                .environmentObject(appState)
                .onAppear {
                    setupApp()
                }
        }
    }
    
    private func setupApp() {
        configureAppearance()
        registerForPushNotifications()
        setupSecurityPolicies()
    }
    
    private func configureAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(named: "PrimaryColor")
        appearance.titleTextAttributes = [.foregroundColor: UIColor.white]
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
    
    private func registerForPushNotifications() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                DispatchQueue.main.async {
                    UIApplication.shared.registerForRemoteNotifications()
                }
            }
        }
    }
    
    private func setupSecurityPolicies() {
        URLCache.shared.removeAllCachedResponses()
        
        HTTPCookieStorage.shared.cookieAcceptPolicy = .never
        
        UserDefaults.standard.set(false, forKey: "NSAllowsArbitraryLoads")
    }
}

class AuthenticationManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var userRole: UserRole = .employee
    
    enum UserRole {
        case employee
        case employer
        case admin
    }
}

class AppState: ObservableObject {
    @Published var selectedTab = 0
    @Published var isLoading = false
    @Published var error: AppError?
}

struct User {
    let id: String
    let email: String
    let firstName: String
    let lastName: String
    let employeeId: String?
    let employerId: String?
}

enum AppError: Error {
    case networkError
    case authenticationError
    case dataError
    case unknown
}