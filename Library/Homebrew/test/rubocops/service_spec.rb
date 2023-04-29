# frozen_string_literal: true

require "rubocops/service"

describe RuboCop::Cop::FormulaAudit::Service do
  subject(:cop) { described_class.new }

  it "reports offenses when a formula's service block uses cellar paths" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        service do
          run [bin/"foo", "run", "-config", etc/"foo/config.json"]
               ^^^ FormulaAudit/Service: Use `opt_bin` instead of `bin` in service blocks.
          working_dir libexec
                      ^^^^^^^ FormulaAudit/Service: Use `opt_libexec` instead of `libexec` in service blocks.
        end
      end
    RUBY

    expect_correction(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        service do
          run [opt_bin/"foo", "run", "-config", etc/"foo/config.json"]
          working_dir opt_libexec
        end
      end
    RUBY
  end

  it "reports no offenses when a formula's service block only uses opt paths" do
    expect_no_offenses(<<~RUBY)
      class Bin < Formula
        url "https://brew.sh/foo-1.0.tgz"

        service do
          run [opt_bin/"bin", "run", "-config", etc/"bin/config.json"]
          working_dir opt_libexec
        end
      end
    RUBY
  end
end
