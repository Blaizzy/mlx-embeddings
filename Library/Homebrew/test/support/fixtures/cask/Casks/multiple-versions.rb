cask "multiple-versions" do
  if MacOS.version == :test_os
    version "1.2.0"
  else
    version "1.2.3"
  end
  sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"

  url "file://#{TEST_FIXTURE_DIR}/cask/caffeine.zip"
  homepage "https://brew.sh/"

  app "Caffeine.app"
end
