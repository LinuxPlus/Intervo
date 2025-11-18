"use client";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Plus, CreditCard } from "lucide-react";
import PaymentMethodCard from "@/components/settings/Card";
import PaymentMethodDialog from "@/components/settings/PaymentMethodDialog";
import SubscriptionConfirmDialog from "@/components/settings/SubscriptionConfirmDialog";
import PayAsYouGoDialog from "@/components/settings/PayAsYouGoDialog";
import { useWorkspace } from "@/context/WorkspaceContext";
import { useToast } from "@/hooks/use-toast";
import Image from "next/image";
import { useSearchParams, useRouter } from "next/navigation";

export default function PaymentMethods() {
  const [open, setOpen] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [showPayAsYouGoDialog, setShowPayAsYouGoDialog] = useState(false);
  const [defaultPmForPaygDialog, setDefaultPmForPaygDialog] = useState(null);
  const [paymentMethods, setPaymentMethods] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPaymentMethod, setSelectedPaymentMethod] = useState(null);
  const { toast } = useToast();
  const searchParams = useSearchParams();
  const router = useRouter();
  const {
    workspaceId,
    getPaymentMethods,
    deletePaymentMethod,
    setDefaultPaymentMethod,
  } = useWorkspace();

  // Check if we're in subscription flow
  const isSubscribing = searchParams.get("subscribe") === "true";
  const isPayAsYouGoFlow = searchParams.get("flow") === "payg";
  const planType = searchParams.get("plan") || "yearly";
  const planPrice = Number(searchParams.get("price")) || 199;
  const priceId = searchParams.get("priceId");

  // Simple plan data based on URL parameters
  const selectedPlan = {
    name: "Business Plan",
    type: planType,
    price: planPrice,
    priceId: priceId,
  };

  const fetchPaymentMethods = async () => {
    if (!workspaceId) {
      console.log(
        "fetchPaymentMethods: workspaceId not available, skipping fetch."
      );
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const result = await getPaymentMethods();

      // Handle both array directly or nested under paymentMethods property
      const methods = Array.isArray(result)
        ? result
        : result.paymentMethods
        ? result.paymentMethods
        : [];

      setPaymentMethods(methods);

      // Determine default payment method for dialogs
      const foundDefaultPm =
        methods.find((m) => m.isDefault) ||
        (methods.length > 0 ? methods[0] : null);

      if (isSubscribing) {
        setSelectedPaymentMethod(foundDefaultPm);
        setShowConfirmDialog(true);
        setShowPayAsYouGoDialog(false);
      } else if (isPayAsYouGoFlow) {
        setDefaultPmForPaygDialog(foundDefaultPm);
        setShowPayAsYouGoDialog(true);
        setShowConfirmDialog(false);
      }
    } catch (error) {
      console.error("Failed to fetch payment methods:", error);
      toast({
        title: error.message || "Failed to load payment methods",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (workspaceId) {
      console.log("Workspace ID available, fetching payment methods...");
      fetchPaymentMethods();
    } else {
      console.log(
        "Waiting for Workspace ID before fetching payment methods..."
      );
      setLoading(true);
    }
  }, [workspaceId]);

  const handleDelete = async (paymentMethodId) => {
    try {
      await deletePaymentMethod(paymentMethodId);
      toast({
        title: "Payment method removed successfully",
        variant: "success",
      });
      // Refresh the list
      fetchPaymentMethods();
    } catch (error) {
      console.error("Failed to delete payment method:", error);
      toast({
        title: error.message || "Failed to remove payment method",
        variant: "destructive",
      });
    }
  };

  const handleSetDefault = async (paymentMethodId) => {
    try {
      await setDefaultPaymentMethod(paymentMethodId);
      toast({
        title: "Default payment method updated",
        variant: "success",
      });
      // Refresh the list
      fetchPaymentMethods();
    } catch (error) {
      console.error("Failed to set default payment method:", error);
      toast({
        title: error.message || "Failed to update default payment method",
        variant: "destructive",
      });
    }
  };

  const handlePaymentMethodAdded = () => {
    toast({
      title: "Payment method added successfully",
      variant: "success",
    });
    fetchPaymentMethods();
    // If we're subscribing and just added a card, show subscription dialog
    if (isSubscribing) {
      setTimeout(() => {
        if (paymentMethods.length > 0) {
          setSelectedPaymentMethod(paymentMethods[0]);
          setShowConfirmDialog(true);
        }
      }, 500);
    }
  };

  const handleSubscriptionConfirmed = async () => {
    try {
      // Simulate API call (replace with real call to your backend)
      await new Promise((resolve) => setTimeout(resolve, 800));

      toast({
        title: "Subscription created successfully!",
        variant: "success",
      });

      // Redirect to billing page without subscription parameters
      router.push(`/${workspaceId}/settings/billing`);
    } catch (error) {
      console.error("Failed to create subscription:", error);
      toast({
        title: error.message || "Failed to create subscription",
        variant: "destructive",
      });
    }
  };

  // --- NEW: Handler for successful PayAsYouGo payment ---
  const handlePayAsYouGoSuccess = (paymentIntentId) => {
    console.log("Pay-as-you-go payment successful, PI:", paymentIntentId);
    toast({
      title: "Credits added successfully!",
      variant: "success",
    });
    // Refresh workspace info potentially, or just navigate away
    // e.g., fetchWorkspaceInfo(); // If credit balance needs update elsewhere
    router.push(`/${workspaceId}/settings/billing`); // Navigate back to billing overview
  };
  // --- END: New handler ---

  // Function to format card brand for display
  const formatCardBrand = (brand) => {
    if (!brand) return "";
    return brand.charAt(0).toUpperCase() + brand.slice(1);
  };

  // Format billing address from payment method details
  const formatBillingAddress = (method) => {
    if (!method.billingDetails || !method.billingDetails.address) {
      return (
        method.billingDetails?.address?.country || "No billing address provided"
      );
    }

    const address = method.billingDetails.address;
    const parts = [];

    if (address.line1) parts.push(address.line1);
    if (address.line2) parts.push(address.line2);
    if (address.city) parts.push(address.city);
    if (address.state) parts.push(address.state);
    if (address.postalCode) parts.push(address.postalCode);
    if (address.country) parts.push(address.country);

    return parts.length > 0 ? parts.join(", ") : "No billing address provided";
  };

  // Loading skeleton for payment methods
  const PaymentMethodSkeleton = () => (
    <div className="animate-pulse">
      <div className="bg-gray-200 h-24 w-full rounded mb-4"></div>
      <div className="bg-gray-200 h-24 w-full rounded"></div>
    </div>
  );

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div className="rounded px-4 py-4">
          <Image
            src="/images/payment/powered-by-stripe.svg"
            alt="Powered by Stripe"
            width={134}
            height={32}
          />
        </div>

        <Button
          onClick={() => setOpen(true)}
          variant="outline"
          className="border-gray-300"
        >
          <Plus className="h-4 w-3" /> Add Card
        </Button>
      </div>

      <div className="space-y-3">
        {loading ? (
          <PaymentMethodSkeleton />
        ) : paymentMethods.length === 0 ? (
          <div className="py-12 text-center border rounded-lg">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
              <CreditCard className="h-6 w-6 text-gray-600" />
            </div>
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              No payment methods
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              Add a payment method to get started with your subscription.
            </p>
            <div className="mt-6">
              <Button onClick={() => setOpen(true)} variant="outline">
                <Plus className="mr-2 h-4 w-4" /> Add Card
              </Button>
            </div>
          </div>
        ) : (
          paymentMethods.map((method) => (
            <PaymentMethodCard
              key={method.id}
              title={`${formatCardBrand(method.brand)} •••• ${method.last4}`}
              brand={formatCardBrand(method.brand)}
              last4={method.last4}
              expMonth={method.expMonth}
              expYear={method.expYear}
              isDefault={method.isDefault}
              billingAddress={formatBillingAddress(method)}
              onSetDefault={() => handleSetDefault(method.id)}
              onDelete={() => handleDelete(method.id)}
            />
          ))
        )}
      </div>

      <PaymentMethodDialog
        open={open}
        onOpenChange={setOpen}
        onSuccess={handlePaymentMethodAdded}
      />

      {showConfirmDialog && (
        <SubscriptionConfirmDialog
          open={showConfirmDialog}
          onOpenChange={setShowConfirmDialog}
          onSuccess={handleSubscriptionConfirmed}
          plan={selectedPlan}
          paymentMethod={selectedPaymentMethod}
        />
      )}

      {showPayAsYouGoDialog && (
        <PayAsYouGoDialog
          open={showPayAsYouGoDialog}
          onOpenChange={(isOpen) => {
            setShowPayAsYouGoDialog(isOpen);
            if (!isOpen) {
              setDefaultPmForPaygDialog(null);
              const params = new URLSearchParams(searchParams);
              params.delete("flow");
              router.replace(`?${params.toString()}`, undefined, {
                shallow: true,
              });
            }
          }}
          onSuccess={handlePayAsYouGoSuccess}
          defaultPaymentMethodFromParent={defaultPmForPaygDialog}
        />
      )}
    </div>
  );
}
