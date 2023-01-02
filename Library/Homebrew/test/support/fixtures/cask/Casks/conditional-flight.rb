cask "conditional-flight" do
  version "1.2.3"
  sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"

  url "file://#{TEST_FIXTURE_DIR}/cask/caffeine/#{platform}/#{version}/#{arch}.zip"
  homepage "https://brew.sh/"

  on_big_sur do
    preflight do
      puts "preflight on Big Sur"
    end
    uninstall_postflight do
      puts "uninstall_postflight on Big Sur"
    end
  end
  on_catalina :or_older do
    preflight do
      puts "preflight on Catalina or older"
    end
  end
end
