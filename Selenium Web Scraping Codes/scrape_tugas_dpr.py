"""
DPR RI Tugas, Wewenang dan Hak Scraper
Scrape informasi dari halaman Tugas, Wewenang dan Hak DPR RI
URL: https://www.dpr.go.id/tentang-dpr/tugas-wewenang-hak
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import csv
import time
from datetime import datetime

class DPRTugasWewenangScraper:
    def __init__(self, headless=False):
        self.setup_driver(headless)
        self.scraped_data = {
            'dpr_sebagai_lembaga_negara': [],
            'anggota_dpr': []
        }
        self.base_url = "https://www.dpr.go.id/tentang-dpr/tugas-wewenang-hak"
        
    def setup_driver(self, headless=False):
        """Setup Chrome WebDriver"""
        print("🤖 Setting up Chrome WebDriver...")
        
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument("--headless")
            
        # Optimization options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.wait = WebDriverWait(self.driver, 30)
            print("✅ WebDriver ready")
        except Exception as e:
            print(f"❌ Failed to setup WebDriver: {e}")
            print("💡 Make sure ChromeDriver is installed")
            raise
    
    def navigate_to_page(self):
        """Navigate to DPR Tugas Wewenang page"""
        print(f"🌐 Navigating to: {self.base_url}")
        
        try:
            self.driver.get(self.base_url)
            
            # Wait for page to load
            print("⏳ Waiting for page to load...")
            
            # Wait for the tab structure to be present
            tab_container = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[role='tablist']"))
            )
            
            print("✅ Tugas Wewenang page loaded")
            time.sleep(3)  # Additional wait for content
            
            return True
            
        except TimeoutException:
            print("❌ Timeout loading page")
            return False
        except Exception as e:
            print(f"❌ Error navigating to page: {e}")
            return False
    
    def get_available_tabs(self):
        """Get all available tabs on the page"""
        print("🔍 Detecting available tabs...")
        
        try:
            # Find tab buttons
            tab_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "[role='tablist'] button[role='tab']"
            )
            
            tabs = []
            for tab in tab_buttons:
                tab_text = tab.text.strip()
                tab_id = tab.get_attribute('id')
                aria_controls = tab.get_attribute('aria-controls')
                is_selected = tab.get_attribute('aria-selected') == 'true'
                
                tabs.append({
                    'text': tab_text,
                    'id': tab_id,
                    'aria_controls': aria_controls,
                    'is_selected': is_selected,
                    'element': tab
                })
            
            print(f"📑 Found {len(tabs)} tabs:")
            for i, tab in enumerate(tabs, 1):
                status = "🔴 Active" if tab['is_selected'] else "⚪ Inactive"
                print(f"   {i}. {tab['text']} {status}")
            
            return tabs
            
        except Exception as e:
            print(f"❌ Error detecting tabs: {e}")
            return []
    
    def click_tab(self, tab_element):
        """Click on a tab to activate it"""
        try:
            # Scroll to tab if needed
            self.driver.execute_script("arguments[0].scrollIntoView(true);", tab_element)
            time.sleep(1)
            
            # Click the tab
            self.driver.execute_script("arguments[0].click();", tab_element)
            time.sleep(2)  # Wait for content to load
            
            return True
            
        except Exception as e:
            print(f"❌ Error clicking tab: {e}")
            return False
    
    def extract_accordion_content(self, tab_name):
        """Extract content from accordion items in current tab"""
        print(f"📋 Extracting accordion content for: {tab_name}")
        
        accordion_data = []
        
        try:
            # Find all accordion buttons in the active tab panel
            accordion_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "[role='tabpanel'][data-headlessui-state='selected'] button[id*='headlessui-disclosure-button']"
            )
            
            print(f"   📁 Found {len(accordion_buttons)} accordion items")
            
            for i, button in enumerate(accordion_buttons, 1):
                try:
                    # Get accordion title
                    title_span = button.find_element(By.CSS_SELECTOR, "span.flex.items-center")
                    title = title_span.text.strip()
                    
                    print(f"   📄 {i}. Processing: {title}")
                    
                    # Check if accordion is already expanded
                    is_expanded = button.get_attribute('aria-expanded') == 'true'
                    
                    # If not expanded, click to expand
                    if not is_expanded:
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        time.sleep(1)
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(2)  # Wait for content to expand
                    
                    # Try to extract content (this might vary based on the actual content structure)
                    accordion_item = {
                        'title': title,
                        'content': '',
                        'expanded': True
                    }
                    
                    # Look for content panel associated with this button
                    try:
                        # Find the parent container and look for content
                        parent_container = button.find_element(By.XPATH, "../..")
                        content_divs = parent_container.find_elements(By.CSS_SELECTOR, "div[id*='headlessui-disclosure-panel']")
                        
                        if content_divs:
                            content_text = ""
                            for content_div in content_divs:
                                if content_div.is_displayed():
                                    content_text += content_div.text.strip() + "\n"
                            accordion_item['content'] = content_text.strip()
                        else:
                            print(f"      ⚠️  No content panel found for: {title}")
                            
                    except NoSuchElementException:
                        print(f"      ⚠️  Content not accessible for: {title}")
                    
                    accordion_data.append(accordion_item)
                    print(f"      ✅ Extracted: {title}")
                    
                    # Collapse back if we expanded it
                    if not is_expanded:
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(1)
                    
                except Exception as e:
                    print(f"      ❌ Error processing accordion item {i}: {e}")
                    continue
            
            print(f"   ✅ Extracted {len(accordion_data)} accordion items")
            return accordion_data
            
        except Exception as e:
            print(f"❌ Error extracting accordion content: {e}")
            return []
    
    def scrape_all_tabs(self):
        """Main scraping function - scrape all tabs"""
        
        print("🚀 Starting DPR Tugas Wewenang scraping...")
        print(f"📅 Scraper started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Navigate to page
            if not self.navigate_to_page():
                print("❌ Failed to load page")
                return {}
            
            # Get available tabs
            tabs = self.get_available_tabs()
            if not tabs:
                print("❌ No tabs found")
                return {}
            
            # Process each tab
            for i, tab in enumerate(tabs, 1):
                print(f"\n📑 === TAB {i}/{len(tabs)}: {tab['text']} ===")
                
                # Click tab to activate it (if not already active)
                if not tab['is_selected']:
                    print(f"🖱️  Clicking tab: {tab['text']}")
                    if not self.click_tab(tab['element']):
                        print(f"❌ Failed to click tab: {tab['text']}")
                        continue
                else:
                    print(f"🔴 Tab already active: {tab['text']}")
                
                # Extract accordion content from this tab
                accordion_content = self.extract_accordion_content(tab['text'])
                
                # Store data based on tab name
                tab_key = self.normalize_tab_name(tab['text'])
                self.scraped_data[tab_key] = accordion_content
                
                print(f"✅ Tab '{tab['text']}' completed: {len(accordion_content)} items")
                
                # Wait between tabs
                time.sleep(2)
            
            print(f"\n🎯 SCRAPING COMPLETED!")
            print(f"📊 Total tabs processed: {len(tabs)}")
            
            # Calculate total items
            total_items = sum(len(content) for content in self.scraped_data.values())
            print(f"📋 Total accordion items: {total_items}")
            
            return self.scraped_data
            
        except KeyboardInterrupt:
            print("\n⚠️  Scraping interrupted by user")
            return self.scraped_data
        except Exception as e:
            print(f"❌ Unexpected error during scraping: {e}")
            return self.scraped_data
        finally:
            self.cleanup()
    
    def normalize_tab_name(self, tab_name):
        """Normalize tab name for use as dictionary key"""
        # Convert to lowercase and replace spaces with underscores
        normalized = tab_name.lower().replace(' ', '_').replace('-', '_')
        
        # Map specific tab names
        mapping = {
            'dpr_sebagai_lembaga_negara': 'dpr_sebagai_lembaga_negara',
            'anggota_dpr': 'anggota_dpr'
        }
        
        return mapping.get(normalized, normalized)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                print("🧹 WebDriver cleaned up")
        except:
            pass
    
    def save_results(self):
        """Save results to files"""
        
        if not self.scraped_data or all(not content for content in self.scraped_data.values()):
            print("❌ No data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON (main format)
        json_file = f"dpr_tugas_wewenang_{timestamp}.json"
        try:
            with open(json_file, 'w', encoding='utf-8') as file:
                json.dump(self.scraped_data, file, indent=2, ensure_ascii=False)
            print(f"💾 JSON saved: {json_file}")
        except Exception as e:
            print(f"❌ JSON save error: {e}")
        
        # Save CSV (flattened format)
        csv_file = f"dpr_tugas_wewenang_{timestamp}.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['tab', 'title', 'content'])
                
                for tab_name, accordion_items in self.scraped_data.items():
                    for item in accordion_items:
                        writer.writerow([
                            tab_name,
                            item.get('title', ''),
                            item.get('content', '').replace('\n', ' ')
                        ])
            print(f"💾 CSV saved: {csv_file}")
        except Exception as e:
            print(f"❌ CSV save error: {e}")
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print detailed summary"""
        
        print(f"\n📊 DETAILED SUMMARY:")
        print(f"=" * 50)
        
        total_items = 0
        
        for tab_name, accordion_items in self.scraped_data.items():
            item_count = len(accordion_items)
            total_items += item_count
            
            print(f"\n📑 {tab_name.replace('_', ' ').title()}: {item_count} items")
            
            for i, item in enumerate(accordion_items, 1):
                title = item.get('title', 'No title')
                content_length = len(item.get('content', ''))
                print(f"   {i}. {title} ({content_length} chars)")
        
        print(f"\n📋 Total accordion items across all tabs: {total_items}")
        
        # Show sample content
        if total_items > 0:
            print(f"\n📝 Sample content:")
            for tab_name, accordion_items in self.scraped_data.items():
                if accordion_items:
                    sample_item = accordion_items[0]
                    print(f"\n🔖 {tab_name}:")
                    print(f"   Title: {sample_item.get('title', 'N/A')}")
                    content = sample_item.get('content', 'N/A')
                    print(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}")

def main():
    print("🏛️  DPR RI TUGAS, WEWENANG DAN HAK SCRAPER")
    print("Scraping information from Tugas, Wewenang dan Hak page")
    print("=" * 65)
    
    # Ask user preference
    print("\n🤖 Choose mode:")
    print("1. Visible browser (you can see what's happening) - Recommended")
    print("2. Headless mode (faster, no browser window)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    headless = choice == "2"
    
    if headless:
        print("⚠️  Headless mode: No browser window will be shown")
    else:
        print("👀 Visible mode: Browser window will open")
    
    scraper = DPRTugasWewenangScraper(headless=headless)
    
    try:
        data = scraper.scrape_all_tabs()
        
        if data and any(content for content in data.values()):
            scraper.save_results()
            print("\n✅ SCRAPING COMPLETED SUCCESSFULLY!")
            
            # Count total items
            total_items = sum(len(content) for content in data.values())
            print(f"🎯 Total items scraped: {total_items}")
        else:
            print("\n❌ No data extracted")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        scraper.save_results()  # Save partial results
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        
    finally:
        try:
            scraper.cleanup()
        except:
            pass
    
    print("\n🏁 Program finished")

if __name__ == "__main__":
    main()