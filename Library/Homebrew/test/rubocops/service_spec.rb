# frozen_string_literal: true

require "rubocops/service"

describe RuboCop::Cop::FormulaAudit::Service do
  subject(:cop) { described_class.new }

  it "reports offenses when a service block is missing a required command" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        service do
        ^^^^^^^^^^ FormulaAudit/Service: Service blocks require `run`, `plist_name` or `service_name` to be defined.
          run_type :cron
          working_dir "/tmp/example"
        end
      end
    RUBY
  end

  it "reports no offenses when a service block only includes custom names" do
    expect_no_offenses(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        service do
          plist_name "custom.mcxl.foo"
          service_name "custom.foo"
        end
      end
    RUBY
  end

  it "reports offenses when a service block includes more than custom names and no run command" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"

        service do
        ^^^^^^^^^^ FormulaAudit/Service: `run` must be defined to use methods other than `service_name` and `plist_name` like [:working_dir].
          plist_name "custom.mcxl.foo"
          service_name "custom.foo"
          working_dir "/tmp/example"
        end
      end
    RUBY
  end

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

  it "reports no offenses when a service block only uses opt paths" do
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
