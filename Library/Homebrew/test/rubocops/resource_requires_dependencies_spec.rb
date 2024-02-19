# frozen_string_literal: true

require "rubocops/resource_requires_dependencies"

describe RuboCop::Cop::FormulaAudit::ResourceRequiresDependencies do
  subject(:cop) { described_class.new }

  context "when a formula does not have the lxml resource" do
    it "does not report offenses" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          uses_from_macos "libxml2"

          resource "not-lxml" do
            url "blah"
            sha256 "blah"
          end
        end
      RUBY
    end
  end

  context "when a formula has the lxml resource" do
    it "does not report offenses if the dependencies are present" do
      expect_no_offenses(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          uses_from_macos "libxml2"
          uses_from_macos "libxslt"

          resource "lxml" do
            url "blah"
            sha256 "blah"
          end
        end
      RUBY
    end

    it "reports offenses if missing a dependency" do
      expect_offense(<<~RUBY)
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          homepage "https://brew.sh"

          uses_from_macos "libsomethingelse"
          uses_from_macos "not_libxml2"

          resource "lxml" do
          ^^^^^^^^^^^^^^^ FormulaAudit/ResourceRequiresDependencies: Add `uses_from_macos` lines above for "libxml2"` and "libxslt".
            url "blah"
            sha256 "blah"
          end
        end
      RUBY
    end
  end
end
