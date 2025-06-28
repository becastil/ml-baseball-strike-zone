import SwiftUI

struct ContentView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        if authManager.isAuthenticated {
            MainTabView()
        } else {
            LoginView()
        }
    }
}

struct MainTabView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        TabView(selection: $appState.selectedTab) {
            if authManager.userRole == .employee {
                EmployeeHomeView()
                    .tabItem {
                        Label("Home", systemImage: "house.fill")
                    }
                    .tag(0)
                
                BenefitsView()
                    .tabItem {
                        Label("Benefits", systemImage: "heart.text.square.fill")
                    }
                    .tag(1)
                
                ClaimsView()
                    .tabItem {
                        Label("Claims", systemImage: "doc.text.fill")
                    }
                    .tag(2)
                
                IDCardView()
                    .tabItem {
                        Label("ID Card", systemImage: "person.crop.rectangle.fill")
                    }
                    .tag(3)
                
                MoreView()
                    .tabItem {
                        Label("More", systemImage: "ellipsis")
                    }
                    .tag(4)
            } else {
                EmployerDashboardView()
                    .tabItem {
                        Label("Dashboard", systemImage: "chart.line.uptrend.xyaxis")
                    }
                    .tag(0)
                
                ReportsView()
                    .tabItem {
                        Label("Reports", systemImage: "chart.bar.doc.horizontal.fill")
                    }
                    .tag(1)
                
                EligibilityView()
                    .tabItem {
                        Label("Eligibility", systemImage: "person.3.fill")
                    }
                    .tag(2)
                
                AnalyticsView()
                    .tabItem {
                        Label("Analytics", systemImage: "chart.pie.fill")
                    }
                    .tag(3)
                
                SettingsView()
                    .tabItem {
                        Label("Settings", systemImage: "gearshape.fill")
                    }
                    .tag(4)
            }
        }
        .accentColor(Color("PrimaryColor"))
    }
}

struct LoginView: View {
    var body: some View {
        Text("Login View - To be implemented")
    }
}

struct EmployeeHomeView: View {
    var body: some View {
        NavigationView {
            Text("Employee Home")
                .navigationTitle("Welcome")
        }
    }
}

struct BenefitsView: View {
    var body: some View {
        NavigationView {
            Text("Benefits View")
                .navigationTitle("My Benefits")
        }
    }
}

struct ClaimsView: View {
    var body: some View {
        NavigationView {
            Text("Claims View")
                .navigationTitle("My Claims")
        }
    }
}

struct IDCardView: View {
    var body: some View {
        NavigationView {
            Text("ID Card View")
                .navigationTitle("ID Card")
        }
    }
}

struct MoreView: View {
    var body: some View {
        NavigationView {
            Text("More Options")
                .navigationTitle("More")
        }
    }
}

struct EmployerDashboardView: View {
    var body: some View {
        NavigationView {
            Text("Employer Dashboard")
                .navigationTitle("Dashboard")
        }
    }
}

struct ReportsView: View {
    var body: some View {
        NavigationView {
            Text("Reports View")
                .navigationTitle("Reports")
        }
    }
}

struct EligibilityView: View {
    var body: some View {
        NavigationView {
            Text("Eligibility Management")
                .navigationTitle("Eligibility")
        }
    }
}

struct AnalyticsView: View {
    var body: some View {
        NavigationView {
            Text("Analytics View")
                .navigationTitle("Analytics")
        }
    }
}

struct SettingsView: View {
    var body: some View {
        NavigationView {
            Text("Settings")
                .navigationTitle("Settings")
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AuthenticationManager())
        .environmentObject(AppState())
}