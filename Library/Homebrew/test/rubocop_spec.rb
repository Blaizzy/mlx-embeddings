# typed: false
# frozen_string_literal: true

require "open3"

describe "RuboCop" do
  context "when calling `rubocop` outside of the Homebrew environment" do
    before do
      ENV.each_key do |key|
        ENV.delete(key) if key.start_with?("HOMEBREW_")
      end

      ENV["XDG_CACHE_HOME"] = (HOMEBREW_CACHE.realpath/"style").to_s
    end

    it "loads all Formula cops without errors" do
      stdout, stderr, status = Open3.capture3(RUBY_PATH, "-W0", "-S", "rubocop", TEST_FIXTURE_DIR/"testball.rb")
      expect(stderr).to be_empty
      expect(stdout).to include("no offenses detected")
      expect(status).to be_a_success
    end
  end
end
