cask "discontinued" do
  version "1.2.3"
  sha256 "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"

  # This cask is used in --online tests, so we use fake URLs to avoid impacting
  # real servers. The URL paths are specific enough that they'll be
  # understandable if they appear in local server logs.
  url "http://localhost/homebrew/test/cask/audit/livecheck/discontinued-#{version}.dmg"
  name "Discontinued"
  desc "Cask for testing discontinued in livecheck"
  homepage "http://localhost/homebrew/test/cask/audit/livecheck/discontinued"

  app "TestCask.app"

  caveats do
    discontinued
  end
end
