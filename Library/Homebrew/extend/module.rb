# typed: false
# frozen_string_literal: true

class Module
  def attr_rw(*attrs)
    attrs.each do |attr|
      module_eval <<-EOS, __FILE__, __LINE__+1
        def #{attr}(val=nil)           # def prefix(val=nil)
          @#{attr} ||= nil             #   @prefix ||= nil
          return @#{attr} if val.nil?  #   return @prefix if val.nil?
          @#{attr} = val               #   @prefix = val
        end                            # end
      EOS
    end
  end
end
