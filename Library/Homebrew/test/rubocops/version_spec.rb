# frozen_string_literal: true

require "rubocops/version"

describe RuboCop::Cop::FormulaAudit::Version do
  subject(:cop) { described_class.new }

  context "When auditing version" do
    it "version should not be an empty string" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          version ""
          ^^^^^^^^^^ version is set to an empty string
        end
      RUBY
    end

    it "version should not have a leading 'v'" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          version "v1.0"
          ^^^^^^^^^^^^^^ version v1.0 should not have a leading 'v'
        end
      RUBY
    end

    it "version should not end with underline and number" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url 'https://brew.sh/foo-1.0.tgz'
          version "1_0"
          ^^^^^^^^^^^^^ version 1_0 should not end with an underline and a number
        end
      RUBY
    end
  end
end
